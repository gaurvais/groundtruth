from __future__ import annotations

import logging
import math
import os
import ssl
from pathlib import Path

import boto3
import urllib3
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

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
S3_BUCKET: str | None = os.getenv("GROUNDTRUTH_S3_BUCKET")
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

def _find_data_file(stem: str) -> Path | None:
    """Return the actual path of a data file if found, otherwise None (lazy resolution)."""
    if not DATA_DIR.exists():
        return None
    candidates = list(DATA_DIR.glob(f"{stem}*"))
    if not candidates:
        return None
    for c in candidates:
        if c.name.startswith(stem):
            return c
    return candidates[0]

# Lazy resolution module variables (None if files are absent, does not crash import)
T0_B04 = _find_data_file("t0_b04")
T0_B08 = _find_data_file("t0_b08")
T1_B04 = _find_data_file("t1_b04")
T1_B08 = _find_data_file("t1_b08")
T0_VISUAL = _find_data_file("t0_visual")
T1_VISUAL = _find_data_file("t1_visual")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def calculate_scanned_ha(bbox: tuple[float, float, float, float]) -> float:
    """Haversine estimate of bbox area in hectares."""
    lon_min, lat_min, lon_max, lat_max = bbox
    mean_lat = (lat_min + lat_max) / 2.0
    width_km = (lon_max - lon_min) * 111.32 * math.cos(math.radians(mean_lat))
    height_km = (lat_max - lat_min) * 110.57
    return width_km * height_km * 100.0


def calculate_raster_scanned_ha(transform, shape: tuple[int, int], crs=None) -> float:
    """Compute the actual scanned area in hectares from raster transform and shape."""
    h, w = shape
    # Corner coordinates in raster CRS: top-left (0,0) and bottom-right (w,h)
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    
    is_projected = False
    if crs is not None:
        try:
            import pyproj
            proj_crs = pyproj.CRS.from_user_input(crs)
            is_projected = proj_crs.is_projected
        except Exception:
            if abs(x0) > 360 or abs(y0) > 360:
                is_projected = True
    elif abs(x0) > 360 or abs(y0) > 360:
        is_projected = True

    if is_projected:
        area_m2 = abs(x1 - x0) * abs(y1 - y0)
        return float(area_m2 / 10000.0)

    # If coordinates are in degrees (EPSG:4326)
    lon_min, lon_max = min(x0, x1), max(x0, x1)
    lat_min, lat_max = min(y0, y1), max(y0, y1)
    return calculate_scanned_ha((lon_min, lat_min, lon_max, lat_max))
