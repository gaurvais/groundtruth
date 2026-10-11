from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import boto3
import numpy as np
import rasterio
from botocore.exceptions import ClientError
from PIL import Image
from pydantic import BaseModel, field_validator
from pystac_client import Client as STACClient
from rasterio.transform import xy as rio_xy
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from utils.config import (
    AWS_REGION,
    BEDROCK_MODEL_ID,
    DATA_DIR,
    STAC_API_URL,
    STAC_COLLECTION,
    T1_VISUAL,
)

log = logging.getLogger(__name__)


class ReviewReport(BaseModel):
    """Structured review report returned by Claude."""
    parcel_id: str
    confidence: float
    violation_type: str
    action: Literal["FLAG_FOR_REVIEW", "SUPPRESS_NO_ACTION"]
    reasoning: str
    model_used: str = ""
    region_used: str = ""
    raw_json: str = ""

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# Alias for backwards compatibility
InterdictionReport = ReviewReport


def _crop_stac_visual(cluster: dict, ndvi_data: dict, scene_id_key: str) -> bytes:
    """Fetch STAC TCI (visual) asset window around the cluster with ~6x padding."""
    scene_id = ndvi_data[scene_id_key]
    catalog = STACClient.open(STAC_API_URL)
    search = catalog.search(collections=[STAC_COLLECTION], ids=[scene_id])
    items = list(search.items())
    if not items:
        raise ValueError(f"STAC scene {scene_id} not found.")
    item = items[0]
    tci_href = item.assets["visual"].href

    # Cluster pixel bounds on NDVI grid
    r_min, c_min, r_max, c_max = cluster["bbox"]
    box_h, box_w = r_max - r_min, c_max - c_min

    # ~6x padding -> add 2.5x to each side
    pad_h, pad_w = int(box_h * 2.5), int(box_w * 2.5)
    r_min, r_max = r_min - pad_h, r_max + pad_h
    c_min, c_max = c_min - pad_w, c_max + pad_w

    # Get spatial bounds in EPSG:4326 using NDVI window transform
    tf = ndvi_data["transform"]
    crs = ndvi_data["crs"]
    left, top = rio_xy(tf, r_min, c_min, offset="ul")
    right, bottom = rio_xy(tf, r_max, c_max, offset="lr")

    lons, lats = warp_transform(crs, "EPSG:4326", [left, right], [top, bottom])
    bbox_lonlat = (min(lons), min(lats), max(lons), max(lats))

    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    with rasterio.Env(**env_opts), rasterio.open(tci_href) as src:
        dst_bounds = transform_bounds("EPSG:4326", src.crs, *bbox_lonlat)
        win = from_bounds(*dst_bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        tci_data = src.read(window=win)

    # TCI is 3-band uint8 (RGB)
    if tci_data.shape[0] >= 3:
        tci_data = tci_data[:3]
    tci_data = np.transpose(tci_data, (1, 2, 0))

    img = Image.fromarray(tci_data, mode="RGB")
    img = img.resize((512, 512), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _crop_visual_image(
    bbox: tuple[int, int, int, int],
    full_shape: tuple[int, int],
    visual_path: Path,
) -> bytes:
    """
    Crop the T2 (Post-Disturbance) visual PNG to the cluster bounding box.
    Returns JPEG bytes suitable for embedding in a Bedrock message.
    """
    row_min, col_min, row_max, col_max = bbox

    with Image.open(visual_path) as img:
        img_w, img_h = img.size
        band_h, band_w = full_shape
        scale_x = img_w / band_w
        scale_y = img_h / band_h

        left = max(0, int(col_min * scale_x))
        top = max(0, int(row_min * scale_y))
        right = min(img_w, int((col_max + 1) * scale_x))
        bottom = min(img_h, int((row_max + 1) * scale_y))

        # Apply a 6x bounding box padding around the cluster footprint
        box_w = right - left
        box_h = bottom - top
        pad_x = int(box_w * 2.5)
        pad_y = int(box_h * 2.5)

        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(img_w, right + pad_x)
        bottom = min(img_h, bottom + pad_y)

        cropped = img.crop((left, top, right, bottom))
        cropped = cropped.resize((512, 512), Image.Resampling.LANCZOS)

        if cropped.mode in ("RGBA", "LA"):
            background = Image.new("RGB", cropped.size, (255, 255, 255))
            background.paste(cropped, mask=cropped.split()[-1])
            cropped = background
        elif cropped.mode != "RGB":
            cropped = cropped.convert("RGB")

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


_VERIFICATION_PROMPT = """\
You are an expert remote-sensing analyst assisting a District Town Planner \
(DTP) in screening suspected ground disturbance and plot carving.

LEGAL & COMPLIANCE DIRECTIVES:
- Do NOT make legal findings or accusations. You must NEVER use the words "illegal" or "unauthorized" in your reasoning or output.
- Explicitly describe any detected disturbance using the phrasing: "suspected clearing, unverified, licence status not checked".
- Any ecological reclamation or plantation recommendations must be explicitly labeled as "indicative".
- Baseline land use is an administrative survey (FMDA survey), not statutory master plan zoning.

Context:
- Location: approximately {lat:.5f}°N, {lon:.5f}°E.
- Baseline date (T0): {t0_date}
- Post-disturbance date (T1): {t1_date}
- Mean NDVI drop detected: {ndvi_drop:.3f}
- Baseline Land Use (FMDA survey): {dominant_class}
- Near Waterbody (<100m): {near_water}

Analysis Guidelines:
1. Examine the provided satellite crop(s). Explicitly weigh whether the NDVI drop represents deliberate anthropogenic ground disturbance (soil scraping, dirt roads, plot subdivision) versus seasonal agricultural harvest, dry-out, or sensor artefacts.
2. Focus on visible physical ground alterations: soil excavation, vegetation loss, and nascent road corridors.
3. Report an honest, calibrated confidence score between 0.0 and 1.0 reflecting the certainty of the visual evidence.
4. Note that loss of vegetative cooling buffer elevates localized Urban Heat Island (UHI) risk. Any mitigation or plantation recommendations must be strictly labeled as "indicative". Do NOT mention light pollution.
5. Keep your reasoning concise and strictly under 80 words.

Return ONLY a JSON object — no preamble, no markdown — matching this exact schema:
{{
  "parcel_id": "{parcel_id}",
  "confidence": <float 0.0 to 1.0>,
  "violation_type": "<e.g. Suspected Land Clearing (Unverified, licence status not checked) | Seasonal Harvest | Natural Drying | Sensor Artifact>",
  "action": "<FLAG_FOR_REVIEW | SUPPRESS_NO_ACTION>",
  "reasoning": "<concise reasoning under 80 words using compliance phrasing: 'suspected clearing, unverified, licence status not checked', labeling any reclamation proposals as indicative, and avoiding the words 'illegal' or 'unauthorized'>"
}}
"""


def call_bedrock_verification(
    cluster: dict,
    ndvi_data: dict,
    parcel_id: str = "FARIDABAD-001",
    model_id: str | None = None,
    region: str | None = None,
) -> ReviewReport:
    """
    Send pre/post crops and prompt to Claude via Bedrock converse API.
    Validates the response against ReviewReport schema and returns it.
    """
    target_model_id = model_id or BEDROCK_MODEL_ID
    target_region = region or AWS_REGION

    raw_class = cluster.get("dominant_class")
    if not raw_class or raw_class in ("No survey coverage", "Unknown"):
        dominant_class = "not available"
    else:
        dominant_class = raw_class

    near_water_val = cluster.get("near_water")
    if near_water_val is True:
        near_water_str = "Yes (<100m)"
    elif near_water_val is False:
        near_water_str = "No"
    else:
        near_water_str = "not available"

    t0_date = ndvi_data.get("t0_date", "baseline")
    t1_date = ndvi_data.get("t1_date", "post-disturbance")

    prompt_text = _VERIFICATION_PROMPT.format(
        lat=cluster["lat"],
        lon=cluster["lon"],
        ndvi_drop=cluster["ndvi_drop_mean"],
        dominant_class=dominant_class,
        near_water=near_water_str,
        parcel_id=parcel_id,
        t0_date=t0_date,
        t1_date=t1_date,
    )

    has_dual_stac = ("scene_t0" in ndvi_data) and ("scene_t1" in ndvi_data)
    image_content: list[dict] = []

    if has_dual_stac:
        try:
            t0_bytes = _crop_stac_visual(cluster, ndvi_data, "scene_t0")
            t1_bytes = _crop_stac_visual(cluster, ndvi_data, "scene_t1")
            image_content = [
                {"image": {"format": "jpeg", "source": {"bytes": t0_bytes}}},
                {"text": f"Image 1: Baseline T0 ({t0_date})"},
                {"image": {"format": "jpeg", "source": {"bytes": t1_bytes}}},
                {"text": f"Image 2: Post-event T1 ({t1_date})\n\n{prompt_text}"},
            ]
        except Exception as exc:
            log.warning("Could not crop dual STAC scenes (%s). Falling back to T1.", exc)
            t1_bytes = _crop_stac_visual(cluster, ndvi_data, "scene_t1")
            image_content = [
                {"image": {"format": "jpeg", "source": {"bytes": t1_bytes}}},
                {"text": prompt_text},
            ]
    elif "scene_t1" in ndvi_data:
        t1_bytes = _crop_stac_visual(cluster, ndvi_data, "scene_t1")
        image_content = [
            {"image": {"format": "jpeg", "source": {"bytes": t1_bytes}}},
            {"text": prompt_text},
        ]
    else:
        t1_path = T1_VISUAL or (DATA_DIR / "t1_visual.png")
        t1_bytes = _crop_visual_image(cluster["bbox"], ndvi_data["shape"], t1_path)
        image_content = [
            {"image": {"format": "jpeg", "source": {"bytes": t1_bytes}}},
            {"text": prompt_text},
        ]

    client = boto3.client("bedrock-runtime", region_name=target_region)
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.converse(
                modelId=target_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": image_content,
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

            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                clean = clean.removeprefix("json")

            report_dict = json.loads(clean)
            report_dict["parcel_id"] = parcel_id
            report = ReviewReport(**report_dict)

            report.model_used = target_model_id
            report.region_used = target_region
            report.raw_json = clean

            return report

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            log.warning("Bedrock ClientError (%s, HTTP %d): %s", error_code, http_status, error_msg)

            # Fail fast on authentication, authorization, or input parameter validation
            if error_code in ("AccessDeniedException", "ValidationException"):
                raise RuntimeError(f"Bedrock Client Error [{error_code}]: {error_msg}")

            # Only retry throttling or 5xx server errors
            is_throttling = error_code in ("ThrottlingException", "ModelTimeoutException")
            is_5xx = (http_status >= 500) or (error_code in ("InternalServerException", "ServiceUnavailableException"))

            if (is_throttling or is_5xx) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.info("Retrying Bedrock call (attempt %d/%d) in %ds...", attempt + 1, max_retries, delay)
                time.sleep(delay)
                continue

            raise RuntimeError(f"Bedrock API Error [{error_code}]: {error_msg}")

        except json.JSONDecodeError as e:
            log.warning("Bedrock returned malformed JSON: %s", e)
            if attempt < max_retries - 1:
                time.sleep(base_delay)
                continue
            raise RuntimeError(f"Bedrock returned malformed JSON after {max_retries} attempts.")
        except Exception as e:
            raise RuntimeError(f"Unexpected Bedrock error: {e!s}")

    raise RuntimeError("Max retries exceeded calling Bedrock.")


def upload_to_s3(
    image_bytes: bytes,
    report: ReviewReport,
    parcel_id: str,
) -> dict[str, str]:
    """Upload the cropped image (.jpg) and JSON report to S3."""
    bucket = os.getenv("GROUNDTRUTH_S3_BUCKET")
    if not bucket:
        raise ValueError("GROUNDTRUTH_S3_BUCKET is not set; S3 upload unavailable.")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    image_key = f"alerts/{parcel_id}/crop.jpg"
    report_key = f"alerts/{parcel_id}/report.json"

    s3.put_object(
        Bucket=bucket,
        Key=image_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    log.info("Uploaded image → s3://%s/%s", bucket, image_key)

    report_bytes = report.model_dump_json(indent=2).encode()
    s3.put_object(
        Bucket=bucket,
        Key=report_key,
        Body=report_bytes,
        ContentType="application/json",
    )
    log.info("Uploaded report → s3://%s/%s", bucket, report_key)

    return {"s3_image_key": image_key, "s3_report_key": report_key}
