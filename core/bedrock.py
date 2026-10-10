from __future__ import annotations
import datetime as dt
import hashlib
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
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
import requests
import urllib3
from requests.adapters import HTTPAdapter
from shapely.geometry import shape, Point, box
from PIL import Image
from pydantic import BaseModel, field_validator
from scipy import ndimage
from dotenv import load_dotenv

try:
    from pystac_client import Client as STACClient
    HAS_PYSTAC = True
except ImportError:
    HAS_PYSTAC = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
from utils.config import *
from core.stac_ingestion import *


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


from pystac_client import Client as STACClient
import io

def _crop_stac_visual(cluster: dict, ndvi_data: dict, scene_id_key: str) -> bytes:
    """
    Fetch STAC TCI (visual) asset window around the cluster with ~6x padding.
    """
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
    
    from rasterio.warp import transform as warp_transform
    lons, lats = warp_transform(crs, "EPSG:4326", [left, right], [top, bottom])
    bbox_lonlat = (min(lons), min(lats), max(lons), max(lats))

    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    with rasterio.Env(**env_opts):
        with rasterio.open(tci_href) as src:
            dst_bounds = transform_bounds("EPSG:4326", src.crs, *bbox_lonlat)
            win = from_bounds(*dst_bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            tci_data = src.read(window=win)

    # TCI is 3-band uint8 (RGB)
    if tci_data.shape[0] >= 3:
        tci_data = tci_data[:3]
    tci_data = np.transpose(tci_data, (1, 2, 0))
    
    img = Image.fromarray(tci_data, mode="RGB")
    # Upscale
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
from {t1_date}, centred at approximately {lat:.5f}°N, {lon:.5f}°E.

The algorithmic pipeline has detected a mean NDVI drop of {ndvi_drop:.3f} \
in this region between {t0_date} and {t1_date}, suggesting rapid vegetation clearing.

Official Baseline Context (FMDA Survey):
- Baseline Land Use (FMDA survey): {dominant_class}
- Near Water: {near_water}


Confidence Scoring Guide: Synthesize both the visual patterns AND the corroborating GIS facts. If linear tracks, grid patterns, or irregular clearing are present on baseline cropland with a severe NDVI drop, assign a calibrated confidence score reflecting the multi-factor evidence (e.g., 65% to 85%), recognizing that 10m Sentinel-2 imagery acts as a low-resolution tripwire for field review.


Environmental Context: Explicitly note in your reasoning that the unauthorized clearing of this baseline agricultural land removes natural cooling, increases future light pollution, and creates a severe Urban Heat Island (UHI) risk if not reclaimed via plantation drives.

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

    if "scene_t1" in ndvi_data:
        image_bytes = _crop_stac_visual(cluster, ndvi_data, "scene_t1")
    else:
        image_bytes = _crop_visual_image(
            cluster["bbox"], ndvi_data["shape"], T1_VISUAL
        )

    dominant_class = cluster.get("dominant_class", "Agriculture Cropland")
    near_water = cluster.get("near_water", False)

    t0_date = ndvi_data.get("t0_date", "baseline")
    t1_date = ndvi_data.get("t1_date", "post-disturbance")
    prompt_text = _VERIFICATION_PROMPT.format(
        lat=cluster["lat"],
        lon=cluster["lon"],
        ndvi_drop=cluster["ndvi_drop_mean"],
        dominant_class=dominant_class,
        near_water=near_water,
        parcel_id=parcel_id,
        t0_date=t0_date,
        t1_date=t1_date,
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


