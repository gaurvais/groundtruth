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
FILL_RATIO_ARTIFACT_THRESHOLD: float = 0.65  # Rectangles this full → artifact
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

# STAC / Sentinel-2 constants
STAC_API_URL: str = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION: str = "sentinel-2-l2a"
SCL_INVALID_CLASSES: set[int] = {0, 1, 3, 8, 9, 10, 11}
DEFAULT_REFLECTANCE_SCALE: float = 0.0001
DEFAULT_REFLECTANCE_OFFSET: float = -0.1
MIN_CLUSTER_AREA_HA: float = 0.5          # 0.5 ha = 50 pixels at 10 m
STAC_SEARCH_DAYS_BACK: int = 45
MIN_VALID_FRACTION: float = 0.80
T0_GAP_DAYS_MIN: int = 5
T0_GAP_DAYS_MAX: int = 30
T0_GAP_IDEAL_DAYS: int = 10
DEFAULT_BBOX_LONLAT: tuple[float, float, float, float] = (77.27633, 28.306005, 77.295899, 28.325805)
USE_STAC: bool = True

# ---------------------------------------------------------------------------
# Target Regions (Bounding Boxes: lon_min, lat_min, lon_max, lat_max)
# ---------------------------------------------------------------------------
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Ballabgarh Agro Belt": (77.28, 28.22, 77.42, 28.34),
    "Aravalli Ridge Corridor": (77.16, 28.28, 77.30, 28.46),
    "Neharpar Urban Ext.": (77.32, 28.32, 77.44, 28.45),
    "Entire Faridabad": (77.15, 28.20, 77.55, 28.52),
}

# Aliases for backwards compatibility
REGIONS["Neharpar Urban Extension"] = REGIONS["Neharpar Urban Ext."]
REGIONS["Entire Faridabad District"] = REGIONS["Entire Faridabad"]

# ---------------------------------------------------------------------------
# Land Use Color Scale (Altair charts & UI legend swatches)
# ---------------------------------------------------------------------------
LAND_USE_COLORS: dict[str, str] = {
    "Agriculture Cropland": "#8BC34A",       # Light green
    "Forest Plantation": "#2E7D32",          # Dark green
    "Forest": "#2E7D32",                     # Dark green
    "Built-up Urban Public Utilities": "#9E9E9E",  # Grey
    "Built-up Rural Household": "#B0BEC5",   # Light slate grey
    "Built-up Rural Vegetation": "#81C784",  # Soft green
    "Built-up": "#9E9E9E",                   # Grey
    "Wasteland Scrub-Open": "#BCAAA4",       # Muted brown/tan
    "Wasteland": "#BCAAA4",                  # Muted brown/tan
    "Waterbody Reservoir/ Tank": "#4FC3F7",  # Light blue
    "Waterbody Canal": "#29B6F6",            # Cyan blue
    "Waterbody Lake/Pond": "#0288D1",        # Deep blue
    "Waterbody": "#4FC3F7",                  # Light blue
    "Grazing Grassland": "#C5E1A5",          # Pale lime green
    "Unknown": "#78909C",                    # Neutral grey
}

DEFAULT_LAND_USE_COLOR: str = "#78909C"

def get_land_use_color(cls_name: str) -> str:
    """Return hex color for a given land use class, matching exact or keyword."""
    if cls_name in LAND_USE_COLORS:
        return LAND_USE_COLORS[cls_name]
    cls_lower = str(cls_name).lower()
    if "agri" in cls_lower or "crop" in cls_lower:
        return "#8BC34A"
    if "forest" in cls_lower or "tree" in cls_lower or "plantation" in cls_lower:
        return "#2E7D32"
    if "built" in cls_lower or "urban" in cls_lower or "settlement" in cls_lower:
        return "#9E9E9E"
    if "water" in cls_lower or "pond" in cls_lower or "lake" in cls_lower or "canal" in cls_lower or "reservoir" in cls_lower:
        return "#4FC3F7"
    if "waste" in cls_lower or "scrub" in cls_lower:
        return "#BCAAA4"
    if "grass" in cls_lower or "grazing" in cls_lower:
        return "#C5E1A5"
    return DEFAULT_LAND_USE_COLOR


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
DATA_DIR = _HERE.parent / "Data"

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




import math
def calculate_scanned_ha(bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    mean_lat = (lat_min + lat_max) / 2.0
    width_km = (lon_max - lon_min) * 111.32 * math.cos(math.radians(mean_lat))
    height_km = (lat_max - lat_min) * 110.57
    return width_km * height_km * 100
