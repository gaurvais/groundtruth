"""
pipeline.py — GroundTruth Detection Pipeline
============================================
Zero Streamlit code here.  All pure geospatial / ML logic.

Responsibilities:
  1. Load Sentinel-2 B04 / B08 GeoTIFFs for T0 and T1.
  2. Compute NDVI and delta-NDVI.
  3. Cluster anomalies with scipy.ndimage.label.
  4. Filter sensor-artifact clusters by fill-ratio.
  5. Call Claude on AWS Bedrock for visual verification of CANDIDATEs.
  6. Upload crops + reports to S3.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Literal

import boto3
import numpy as np
import rasterio
from rasterio.transform import xy as rio_xy
from PIL import Image
from pydantic import BaseModel, field_validator
from scipy import ndimage

# ---------------------------------------------------------------------------
# Constants — tune these without touching logic
# ---------------------------------------------------------------------------
NDVI_THRESHOLD: float = 0.15          # Minimum delta-NDVI to flag a pixel
FILL_RATIO_ARTIFACT_THRESHOLD: float = 0.58  # Rectangles this full → artifact
MIN_CLUSTER_PIXELS: int = 50          # Ignore tiny specks < N pixels
S3_BUCKET: str = os.getenv("GROUNDTRUTH_S3_BUCKET", "groundtruth-enforcement")
AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")

# Default model ID / inference profile ID for Claude Sonnet 4.6 in ap-southeast-2
BEDROCK_MODEL_ID: str = os.getenv(
    "GROUNDTRUTH_BEDROCK_MODEL",
    "au.anthropic.claude-sonnet-4-6",
)


def list_available_inference_profiles(region: str | None = None) -> list[dict]:
    """
    Discover system-defined and application inference profiles in the given AWS region.
    Returns a list of dicts: [{'id': ..., 'name': ..., 'arn': ...}].
    """
    reg = region or AWS_REGION
    try:
        client = boto3.client("bedrock", region_name=reg)
        resp = client.list_inference_profiles(maxResults=100)
        summaries = resp.get("inferenceProfileSummaries", [])
        profiles = []
        for p in summaries:
            pid = p.get("inferenceProfileId", "")
            pname = p.get("inferenceProfileName", pid)
            parn = p.get("inferenceProfileArn", "")
            if any(term in pid.lower() or term in pname.lower() for term in ["anthropic", "claude", "sonnet"]):
                profiles.append({"id": pid, "name": pname, "arn": parn})
        return profiles
    except Exception as exc:
        log.warning("Could not auto-list inference profiles in %s: %s", reg, exc)
        return []

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the script is portable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DATA_DIR = _HERE / "Data"

# Files may have doubled extensions from the upload; resolve robustly
def _find_data_file(stem: str) -> Path:
    """Return the actual path of a data file, handling doubled extensions."""
    candidates = list(DATA_DIR.glob(f"{stem}*"))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find '{stem}' in {DATA_DIR}. "
            f"Available: {[p.name for p in DATA_DIR.iterdir()]}"
        )
    # Prefer exact stem match, then fallback to first candidate
    for c in candidates:
        if c.name.startswith(stem):
            return c
    return candidates[0]

T0_B04 = _find_data_file("t0_b04")
T0_B08 = _find_data_file("t0_b08")
T1_B04 = _find_data_file("t1_b04")
T1_B08 = _find_data_file("t1_b08")
T0_VISUAL = _find_data_file("t0_visual")
T1_VISUAL = _find_data_file("t1_visual")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# TASK 1 — NDVI + Anomaly Detection
# ===========================================================================

def _load_band(path: Path) -> tuple[np.ndarray, rasterio.transform.Affine, object]:
    """Load a single-band 16-bit GeoTIFF; return (array float32, transform, crs)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI safely; pixels where NIR+Red==0 become NaN."""
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = np.where(
        (nir + red) == 0,
        np.nan,
        (nir - red) / (nir + red),
    )
    return np.nan_to_num(ndvi, nan=0.0).astype(np.float32)


def load_and_compute_ndvi() -> dict:
    """
    Load both epochs, compute NDVI, delta, mask.

    Returns a dict with keys:
        ndvi_t0, ndvi_t1, delta_ndvi, anomaly_mask,
        transform, crs, shape
    """
    red_t0, transform, crs = _load_band(T0_B04)
    nir_t0, _, _ = _load_band(T0_B08)
    red_t1, _, _ = _load_band(T1_B04)
    nir_t1, _, _ = _load_band(T1_B08)

    ndvi_t0 = compute_ndvi(red_t0, nir_t0)
    ndvi_t1 = compute_ndvi(red_t1, nir_t1)

    delta_ndvi = ndvi_t0 - ndvi_t1          # positive = vegetation loss
    anomaly_mask = (delta_ndvi > NDVI_THRESHOLD).astype(np.uint8)

    log.info(
        "NDVI computed. T0 mean=%.3f, T1 mean=%.3f, flagged pixels=%d",
        np.nanmean(ndvi_t0), np.nanmean(ndvi_t1), anomaly_mask.sum(),
    )
    return {
        "ndvi_t0": ndvi_t0,
        "ndvi_t1": ndvi_t1,
        "delta_ndvi": delta_ndvi,
        "anomaly_mask": anomaly_mask,
        "transform": transform,
        "crs": crs,
        "shape": ndvi_t0.shape,
    }


# ===========================================================================
# TASK 2 — Artifact Filtering + Cluster Extraction
# ===========================================================================

def is_artifact_shaped(cluster_mask: np.ndarray) -> bool:
    """
    Return True if the cluster looks like a sensor swath-edge artifact.

    Heuristic: if the flagged pixels fill ≥ FILL_RATIO_ARTIFACT_THRESHOLD
    of their own bounding box they form a near-perfect rectangle — a
    hallmark of nodata stripes rather than real ground disturbance.
    """
    rows = np.any(cluster_mask, axis=1)
    cols = np.any(cluster_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    bbox_area = (row_max - row_min + 1) * (col_max - col_min + 1)
    pixel_count = int(cluster_mask.sum())
    fill_ratio = pixel_count / bbox_area if bbox_area > 0 else 0.0
    return fill_ratio >= FILL_RATIO_ARTIFACT_THRESHOLD


def extract_clusters(ndvi_data: dict) -> list[dict]:
    """
    Label connected components of the anomaly mask.

    For each component big enough (≥ MIN_CLUSTER_PIXELS):
      - compute bounding box in pixel coords
      - convert centroid to lat/lon via rasterio.transform.xy
      - compute fill_ratio and decide status: CANDIDATE or SUPPRESSED

    Returns a list of cluster dicts:
      {id, bbox, lat, lon, ndvi_drop_mean, pixel_count, fill_ratio, status}
    """
    mask = ndvi_data["anomaly_mask"]
    delta = ndvi_data["delta_ndvi"]
    transform = ndvi_data["transform"]

    labeled_array, n_labels = ndimage.label(mask)
    log.info("Found %d raw connected components.", n_labels)

    clusters = []
    for label_id in range(1, n_labels + 1):
        comp_mask = labeled_array == label_id
        pixel_count = int(comp_mask.sum())
        if pixel_count < MIN_CLUSTER_PIXELS:
            continue  # too small — ignore

        # Bounding box
        rows = np.where(np.any(comp_mask, axis=1))[0]
        cols = np.where(np.any(comp_mask, axis=0))[0]
        row_min, row_max = int(rows[0]), int(rows[-1])
        col_min, col_max = int(cols[0]), int(cols[-1])

        bbox_area = (row_max - row_min + 1) * (col_max - col_min + 1)
        fill_ratio = round(pixel_count / bbox_area, 4) if bbox_area > 0 else 0.0

        # Centroid → lat / lon
        centroid_row = (row_min + row_max) // 2
        centroid_col = (col_min + col_max) // 2
        lon, lat = rio_xy(transform, centroid_row, centroid_col)

        # Mean NDVI drop inside cluster
        ndvi_drop_mean = float(np.mean(delta[comp_mask]))

        # Artifact check
        artifact = is_artifact_shaped(comp_mask)
        status = "SUPPRESSED" if artifact else "CANDIDATE"

        clusters.append({
            "id": label_id,
            "bbox": (row_min, col_min, row_max, col_max),
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "ndvi_drop_mean": round(ndvi_drop_mean, 4),
            "pixel_count": pixel_count,
            "fill_ratio": fill_ratio,
            "status": status,
        })

    log.info(
        "Clusters after size filter: %d  (%d CANDIDATE, %d SUPPRESSED)",
        len(clusters),
        sum(1 for c in clusters if c["status"] == "CANDIDATE"),
        sum(1 for c in clusters if c["status"] == "SUPPRESSED"),
    )
    return clusters


# ===========================================================================
# TASK 3 — Pydantic Model + Bedrock Verification
# ===========================================================================

class InterdictionReport(BaseModel):
    """Structured enforcement report returned by Claude."""
    parcel_id: str
    confidence: float
    violation_type: str
    action: Literal["FLAG_FOR_REVIEW", "SUPPRESS_NO_ACTION"]
    reasoning: str

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


def _crop_visual_image(
    bbox: tuple[int, int, int, int],
    full_shape: tuple[int, int],
    visual_path: Path,
) -> bytes:
    """
    Crop the T1 visual PNG to the cluster bounding box (pixel coords).
    Returns JPEG bytes suitable for embedding in a Bedrock message.
    """
    row_min, col_min, row_max, col_max = bbox

    with Image.open(visual_path) as img:
        img_w, img_h = img.size
        # Scale pixel coords from band resolution to PNG resolution
        band_h, band_w = full_shape
        scale_x = img_w / band_w
        scale_y = img_h / band_h

        left   = max(0, int(col_min * scale_x))
        top    = max(0, int(row_min * scale_y))
        right  = min(img_w, int((col_max + 1) * scale_x))
        bottom = min(img_h, int((row_max + 1) * scale_y))

        # Give a little context margin (20 %)
        pad_x = max(10, int((right - left) * 0.2))
        pad_y = max(10, int((bottom - top) * 0.2))
        left   = max(0, left - pad_x)
        top    = max(0, top - pad_y)
        right  = min(img_w, right + pad_x)
        bottom = min(img_h, bottom + pad_y)

        cropped = img.crop((left, top, right, bottom))

        # JPEG does not support alpha channels (RGBA) or palette modes (P).
        # Composite onto a white RGB background to preserve visual appearance.
        if cropped.mode in ("RGBA", "LA"):
            background = Image.new("RGB", cropped.size, (255, 255, 255))
            background.paste(cropped, mask=cropped.split()[-1])  # use alpha as mask
            cropped = background
        elif cropped.mode != "RGB":
            cropped = cropped.convert("RGB")

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


_VERIFICATION_PROMPT = """\
You are an expert remote-sensing analyst assisting a District Town Planner \
(DTP) in identifying unauthorized land development on agricultural land.

The image shows a cropped region of a Sentinel-2 true-colour composite \
from May 2025, centred at approximately {lat:.5f}°N, {lon:.5f}°E.

The algorithmic pipeline has detected a mean NDVI drop of {ndvi_drop:.3f} \
in this region since December 2024, suggesting rapid vegetation clearing.

Your task:
1. Visually inspect the image for signs of unauthorized construction or \
land carving: dirt road grids, boundary wall foundations, soil compaction \
patterns, plot subdivision markings.
2. Distinguish these from natural causes (seasonal dry-out, harvested \
fields) or sensor artefacts.
3. Return ONLY a JSON object — no preamble, no markdown — matching this \
exact schema:
{{
  "parcel_id": "{parcel_id}",
  "confidence": <float 0–1>,
  "violation_type": "<e.g. Unauthorized Plot Carving | Boundary Wall Construction | Agricultural Burn | Natural Drying | Sensor Artifact>",
  "action": "<FLAG_FOR_REVIEW | SUPPRESS_NO_ACTION>",
  "reasoning": "<two or three sentences>"
}}

Be conservative: flag only if you see unambiguous anthropogenic patterns.
"""


def call_bedrock_verification(
    cluster: dict,
    ndvi_data: dict,
    parcel_id: str = "FARIDABAD-001",
    model_id: str | None = None,
    region: str | None = None,
) -> InterdictionReport:
    """
    Crop the T1 visual around *cluster*, send to Claude via Bedrock converse,
    validate the response against InterdictionReport schema, and return it.

    Only call for CANDIDATE clusters.
    """
    if cluster["status"] == "SUPPRESSED":
        raise ValueError(
            f"Cluster {cluster['id']} is SUPPRESSED — Bedrock call skipped."
        )

    target_model_id = model_id or BEDROCK_MODEL_ID
    target_region = region or AWS_REGION

    image_bytes = _crop_visual_image(
        cluster["bbox"], ndvi_data["shape"], T1_VISUAL
    )

    prompt_text = _VERIFICATION_PROMPT.format(
        lat=cluster["lat"],
        lon=cluster["lon"],
        ndvi_drop=cluster["ndvi_drop_mean"],
        parcel_id=parcel_id,
    )

    client = boto3.client("bedrock-runtime", region_name=target_region)

    response = client.converse(
        modelId=target_model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": image_bytes},
                        }
                    },
                    {"text": prompt_text},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )

    raw_text: str = (
        response.get("output", {})
        .get("message", {})
        .get("content", [{}])[0]
        .get("text", "")
    )

    log.info("Bedrock raw response: %s", raw_text[:300])

    # Strip optional markdown fences
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]

    report_dict = json.loads(clean)
    report_dict["parcel_id"] = parcel_id        # enforce correct parcel_id
    report = InterdictionReport(**report_dict)
    return report


# ===========================================================================
# TASK 4 — S3 Upload
# ===========================================================================

def upload_to_s3(
    image_bytes: bytes,
    report: InterdictionReport,
    parcel_id: str,
) -> dict[str, str]:
    """
    Upload the cropped image and JSON report to S3.

    Returns dict with 's3_image_key' and 's3_report_key'.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)

    image_key = f"alerts/{parcel_id}/crop.png"
    report_key = f"alerts/{parcel_id}/report.json"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=image_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    log.info("Uploaded image → s3://%s/%s", S3_BUCKET, image_key)

    report_bytes = report.model_dump_json(indent=2).encode()
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=report_key,
        Body=report_bytes,
        ContentType="application/json",
    )
    log.info("Uploaded report → s3://%s/%s", S3_BUCKET, report_key)

    return {"s3_image_key": image_key, "s3_report_key": report_key}


# ===========================================================================
# Convenience: run the full detection pass (no AI, no S3)
# ===========================================================================

def run_detection() -> tuple[dict, list[dict]]:
    """
    Entry point used by app.py.

    Returns:
        ndvi_data  — raw arrays + transform
        clusters   — list of cluster dicts with status tags
    """
    ndvi_data = load_and_compute_ndvi()
    clusters = extract_clusters(ndvi_data)
    return ndvi_data, clusters


if __name__ == "__main__":
    data, clus = run_detection()
    for c in clus:
        print(c)
