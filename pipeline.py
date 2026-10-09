
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
import ssl
from pathlib import Path
from typing import Literal

import boto3
import numpy as np
import rasterio
from rasterio.transform import xy as rio_xy
import requests
import urllib3
from requests.adapters import HTTPAdapter
from shapely.geometry import shape, Point, box
from PIL import Image
from pydantic import BaseModel, field_validator
from scipy import ndimage
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env if present
load_dotenv()


class LegacyRenegotiationAdapter(HTTPAdapter):
    """
    HTTPAdapter to handle OpenSSL 3.0+ legacy renegotiation restrictions
    frequently encountered on state government GIS endpoints (GMDA/FMDA).
    """
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


# ---------------------------------------------------------------------------
# Constants — tune these without touching logic
# ---------------------------------------------------------------------------
NDVI_THRESHOLD: float = 0.15          # Minimum delta-NDVI to flag a pixel
FILL_RATIO_ARTIFACT_THRESHOLD: float = 0.58  # Rectangles this full → artifact
MIN_CLUSTER_PIXELS: int = 50          # Ignore tiny specks < N pixels
S3_BUCKET: str = os.getenv("GROUNDTRUTH_S3_BUCKET", "groundtruth-enforcement")
AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-southeast-2"

FMDA_LAND_USE_QUERY_URL: str = (
    "https://onemapdepts.gmda.gov.in/server1/rest/services/FMDA/FMDA_Land_Use/MapServer/1/query"
)
FMDA_MASTER_PLAN_EXPORT_URL: str = (
    "https://onemapdepts.gmda.gov.in/server1/rest/services/FMDA/FMDA_MasterPlan2031/MapServer/export"
)

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
# Baseline Land-Use Ingestion & Spatial Enrichment (FMDA GIS)
# ===========================================================================

LAND_USE_CACHE_FILE: Path = DATA_DIR / "land_use.geojson"


def fetch_and_cache_land_use(
    bbox: tuple[float, float, float, float] | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Fetch vector land-use polygons covering our Faridabad tile from the
    official FMDA Land Use ArcGIS REST API:
    https://onemapdepts.gmda.gov.in/server1/rest/services/FMDA/FMDA_Land_Use/MapServer/1/query

    Query using tile bounding box (inSR=4326, outSR=4326, f=geojson,
    outFields=Level1_des, where=1=1, returnGeometry=true), and cache locally
    as data/land_use.geojson. Uses verify=False + LegacyRenegotiationAdapter.
    """
    target_path = output_path or LAND_USE_CACHE_FILE
    if target_path.exists():
        try:
            log.info("Loading cached baseline land-use GeoJSON from %s", target_path)
            return json.loads(target_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not parse cached land-use GeoJSON: %s. Re-fetching.", exc)

    session = requests.Session()
    session.mount("https://", LegacyRenegotiationAdapter())

    # Default tile bounding box in EPSG:4326 (min_lon, min_lat, max_lon, max_lat)
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        min_lon, min_lat, max_lon, max_lat = 77.275, 28.305, 77.297, 28.327

    params = {
        "where": "1=1",
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outSR": "4326",
        "outFields": "Level1_des",
        "returnGeometry": "true",
        "f": "geojson",
    }

    log.info("Querying FMDA MapServer land-use layer for bbox [%s]...", params["geometry"])
    resp = session.get(FMDA_LAND_USE_QUERY_URL, params=params, verify=False, timeout=30)
    resp.raise_for_status()
    geojson_data = resp.json()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")
    log.info("Saved %d land-use features to %s", len(geojson_data.get("features", [])), target_path)
    return geojson_data


def fetch_transparent_master_plan(
    bbox: tuple[float, float, float, float] | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Fetch the Master Plan export image, apply a server-side alpha mask to turn 
    white background pixels transparent, and save locally.
    """
    target_path = output_path or (DATA_DIR / "master_plan_transparent.png")
    if target_path.exists():
        log.info("Loading cached transparent master plan from %s", target_path)
        return target_path

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        min_lon, min_lat, max_lon, max_lat = 77.275, 28.305, 77.297, 28.327

    url = f"https://onemapdepts.gmda.gov.in/server1/rest/services/FMDA/FMDA_MasterPlan2031/MapServer/export?bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=4326&size=1400,1400&layers=show:0&format=png32&f=image"
    
    session = requests.Session()
    session.mount("https://", LegacyRenegotiationAdapter())
    
    log.info("Fetching Master Plan image for alpha masking...")
    resp = session.get(url, verify=False, timeout=30)
    resp.raise_for_status()
    
    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    arr = np.array(img)
    
    # Soft alpha mask: keeps tinted fills while removing pure white/light paper background
    min_rgb = np.min(arr[:, :, :3], axis=2)
    arr[:, :, 3] = np.clip((255 - min_rgb) * 3.5, 0, 255).astype(np.uint8)
    
    transparent_img = Image.fromarray(arr)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    transparent_img.save(target_path, format="PNG")
    
    return target_path

def enrich_clusters_with_land_use(
    clusters: list[dict],
    ndvi_data: dict,
    geojson_data: dict | None = None,
) -> list[dict]:
    """
    For each detected disturbance cluster:
      1. Spatial intersection with FMDA vector polygons to compute its dominant
         baseline land use from Level1_des (e.g. 'Agriculture Cropland', 'Forest',
         'Waterbody Canal').
      2. Flag if within 100m of a waterbody (near_water = True).
      3. Compute priority score based on cluster area, mean NDVI drop, near_water,
         and baseline zoning severity.
    """
    if geojson_data is None:
        geojson_data = fetch_and_cache_land_use()

    features = geojson_data.get("features", [])
    polygons: list[tuple[object, str]] = []
    water_polys: list[object] = []

    for f in features:
        try:
            geom = shape(f["geometry"])
            cls_name = f.get("properties", {}).get("Level1_des", "Unclassified")
            polygons.append((geom, cls_name))
            if "water" in cls_name.lower():
                water_polys.append(geom)
        except Exception as exc:
            log.warning("Skipping feature geometry: %s", exc)

    transform = ndvi_data["transform"]

    for c in clusters:
        r_min, c_min, r_max, c_max = c["bbox"]
        lon1, lat1 = rio_xy(transform, r_min, c_min)
        lon2, lat2 = rio_xy(transform, r_max, c_max)
        c_box = box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        c_point = Point(c["lon"], c["lat"])

        # Spatial intersection for dominant baseline land use
        intersections: dict[str, float] = {}
        for geom, cls_name in polygons:
            if c_box.intersects(geom):
                inter_area = c_box.intersection(geom).area
                intersections[cls_name] = intersections.get(cls_name, 0.0) + inter_area

        if intersections:
            dominant_class = max(intersections, key=intersections.get)
        else:
            containing = [cls_name for geom, cls_name in polygons if geom.contains(c_point)]
            dominant_class = containing[0] if containing else "Agriculture Cropland"

        # Waterbody proximity (< 100m buffer)
        # At latitude ~28.3°N, 1 degree ~ 111,000 meters
        near_water = False
        min_water_dist_m = 999999.0
        for wg in water_polys:
            dist_m = c_point.distance(wg) * 111000.0
            if dist_m < min_water_dist_m:
                min_water_dist_m = dist_m
            if dist_m <= 100.0:
                near_water = True

        # Priority scoring:
        # - Cluster area factor: larger clearing = higher impact (0 to 35 pts)
        # - NDVI drop severity: higher drop = more severe clearing (0 to 35 pts)
        # - Waterbody buffer violation (<100m violates canal/lake buffers): +20 pts
        # - Agricultural/forest conversion: +10 pts
        pix = c["pixel_count"]
        ndvi_drop = c["ndvi_drop_mean"]
        area_norm = min(1.0, pix / 1000.0)
        ndvi_norm = min(1.0, max(0.0, (ndvi_drop - 0.15) / 0.35))
        water_bonus = 20.0 if near_water else 0.0
        agri_bonus = 10.0 if any(k in dominant_class.lower() for k in ["agri", "forest", "crop"]) else 0.0

        priority_score = round((area_norm * 35.0) + (ndvi_norm * 35.0) + water_bonus + agri_bonus, 1)

        c["dominant_class"] = dominant_class
        c["near_water"] = near_water
        c["water_dist_m"] = round(min_water_dist_m, 1)
        c["priority_score"] = priority_score

    # Sort candidates by priority score descending
    candidates = [c for c in clusters if c["status"] == "CANDIDATE"]
    candidates.sort(key=lambda x: x["priority_score"], reverse=True)
    suppressed = [c for c in clusters if c["status"] == "SUPPRESSED"]

    return candidates + suppressed


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
    Crop the T2 (Post-Disturbance) visual PNG to the cluster bounding box (pixel coords).
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

        # Apply a 6x bounding box padding around the cluster footprint
        box_w = right - left
        box_h = bottom - top
        pad_x = int(box_w * 2.5)
        pad_y = int(box_h * 2.5)
        
        left   = max(0, left - pad_x)
        top    = max(0, top - pad_y)
        right  = min(img_w, right + pad_x)
        bottom = min(img_h, bottom + pad_y)

        cropped = img.crop((left, top, right, bottom))
        
        # Upscale to 512x512 using Lanczos resampling
        cropped = cropped.resize((512, 512), Image.Resampling.LANCZOS)

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

Official Baseline Context (FMDA Survey):
- Baseline Land Use (FMDA survey): {dominant_class}
- Near Water: {near_water}

Your task:
1. The parcel has a baseline classification of Agriculture Cropland per the FMDA survey. \
Inspect for active ground clearance, dirt road corridors, and rectangular soil compaction adjacent to settlements. \
Flag as FLAG_FOR_REVIEW if deliberate anthropogenic clearing or road carving is evident on agricultural land, \
even if formal concrete foundations are not yet visible.
2. Consider the previous agricultural/vegetative baseline when evaluating \
disturbance legitimacy. Baseline land use is an administrative survey, not statutory master plan zoning. Do not make definitive legal conclusions. Focus on visible physical ground alterations: soil excavation, vegetation loss, and nascent road corridors. Clearings on designated agricultural cropland or \
within waterbody buffer zones (<100m) represent severe regulatory violations.
3. Distinguish these from natural causes (seasonal dry-out, harvested \
fields) or sensor artefacts.
4. Return ONLY a JSON object — no preamble, no markdown — matching this \
exact schema:
{{
  "parcel_id": "{parcel_id}",
  "confidence": <float 0–1>,
  "violation_type": "<e.g. Suspected Land Clearing (Unverified) | Agricultural Burn | Natural Drying | Sensor Artifact>",
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
    Crop the T2 (Post-Disturbance) visual around *cluster*, send to Claude via Bedrock converse,
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

    dominant_class = cluster.get("dominant_class", "Agriculture Cropland")
    near_water = cluster.get("near_water", False)

    prompt_text = _VERIFICATION_PROMPT.format(
        lat=cluster["lat"],
        lon=cluster["lon"],
        ndvi_drop=cluster["ndvi_drop_mean"],
        dominant_class=dominant_class,
        near_water=near_water,
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
        inferenceConfig={"temperature": 0.0, "maxTokens": 600},
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
        clusters   — list of cluster dicts enriched with baseline land use & priority score
    """
    ndvi_data = load_and_compute_ndvi()
    clusters = extract_clusters(ndvi_data)
    clusters = enrich_clusters_with_land_use(clusters, ndvi_data)
    return ndvi_data, clusters


if __name__ == "__main__":
    data, clus = run_detection()
    for c in clus:
        print(c)
