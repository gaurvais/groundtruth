from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import urllib3
from PIL import Image
from rasterio.transform import xy as rio_xy
from rasterio.warp import transform as warp_transform
from shapely.geometry import Point, box, shape

from utils.config import (
    DATA_DIR,
    FMDA_LAND_USE_QUERY_URL,
    LegacyRenegotiationAdapter,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
log = logging.getLogger(__name__)


LAND_USE_CACHE_FILE: Path = DATA_DIR / "land_use.geojson"


def fetch_and_cache_land_use(
    bbox: tuple[float, float, float, float] | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Fetch vector land-use polygons covering scan bbox from the official FMDA API.
    Caches per bbox hash in Data/cache/land_use_<hash>.geojson.
    """
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_hash = hashlib.md5(f"{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}".encode()).hexdigest()[:10]
        target_path = output_path or (DATA_DIR / "cache" / f"land_use_{bbox_hash}.geojson")
    else:
        min_lon, min_lat, max_lon, max_lat = 77.275, 28.305, 77.297, 28.327
        target_path = output_path or LAND_USE_CACHE_FILE

    if target_path.exists():
        try:
            log.info("Loading cached baseline land-use GeoJSON from %s", target_path)
            return json.loads(target_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not parse cached land-use GeoJSON: %s. Re-fetching.", exc)

    session = requests.Session()
    session.mount("https://", LegacyRenegotiationAdapter())

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

    try:
        log.info("Querying FMDA MapServer land-use layer for bbox [%s]...", params["geometry"])
        resp = session.get(FMDA_LAND_USE_QUERY_URL, params=params, verify=False, timeout=20)
        resp.raise_for_status()
        geojson_data = resp.json()
        if geojson_data.get("features"):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")
            log.info("Saved %d land-use features to %s", len(geojson_data["features"]), target_path)
            return geojson_data
    except Exception as exc:
        log.warning("FMDA land-use query failed (%s). Checking fallback cache.", exc)

    if LAND_USE_CACHE_FILE.exists():
        try:
            log.info("Falling back to bundled baseline land-use: %s", LAND_USE_CACHE_FILE)
            return json.loads(LAND_USE_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {"type": "FeatureCollection", "features": []}


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
      1. Spatial intersection with FMDA vector polygons to compute dominant baseline land use.
      2. Reproject cluster corner coordinates to EPSG:4326 using ndvi_data["crs"].
      3. If no polygon covers cluster, set dominant_class = 'No survey coverage' and near_water = None.
      4. Return ALL clusters (every status), sorted by priority score descending.
    """
    transform = ndvi_data["transform"]
    crs = ndvi_data.get("crs")
    h, w = ndvi_data["shape"]

    if geojson_data is None:
        x0, y0 = rio_xy(transform, 0, 0)
        x1, y1 = rio_xy(transform, h, w)
        if crs and str(crs) not in ("EPSG:4326", "4326"):
            from rasterio.warp import transform as rio_warp_transform
            lons, lats = rio_warp_transform(crs, "EPSG:4326", [x0, x1], [y0, y1])
            scan_bbox = (min(lons), min(lats), max(lons), max(lats))
        else:
            scan_bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        geojson_data = fetch_and_cache_land_use(bbox=scan_bbox)

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

    # Batch reproject all cluster corners
    if clusters:
        x1s, y1s, x2s, y2s = [], [], [], []
        for c in clusters:
            r_min, c_min, r_max, c_max = c["bbox"]
            x1, y1 = rio_xy(transform, r_min, c_min)
            x2, y2 = rio_xy(transform, r_max, c_max)
            x1s.append(x1)
            y1s.append(y1)
            x2s.append(x2)
            y2s.append(y2)

        if crs and str(crs) not in ("EPSG:4326", "4326"):
            all_xs = x1s + x2s
            all_ys = y1s + y2s
            all_lons, all_lats = warp_transform(crs, "EPSG:4326", all_xs, all_ys)
            n = len(clusters)
            lon1s, lon2s = all_lons[:n], all_lons[n:]
            lat1s, lat2s = all_lats[:n], all_lats[n:]
        else:
            lon1s, lon2s = x1s, x2s
            lat1s, lat2s = y1s, y2s
    else:
        lon1s, lon2s, lat1s, lat2s = [], [], [], []

    for idx, c in enumerate(clusters):
        lon1, lon2 = lon1s[idx], lon2s[idx]
        lat1, lat2 = lat1s[idx], lat2s[idx]

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
            dominant_class = containing[0] if containing else "No survey coverage"

        # Waterbody proximity (< 100m buffer)
        if dominant_class == "No survey coverage":
            near_water = None
            min_water_dist_m = 999999.0
        else:
            near_water = False
            min_water_dist_m = 999999.0
            for wg in water_polys:
                dist_m = c_point.distance(wg) * 111000.0
                min_water_dist_m = min(min_water_dist_m, dist_m)
                if dist_m <= 100.0:
                    near_water = True

        # Priority scoring:
        pix = c["pixel_count"]
        ndvi_drop = c["ndvi_drop_mean"]
        area_norm = min(1.0, pix / 1000.0)
        ndvi_norm = min(1.0, max(0.0, (ndvi_drop - 0.15) / 0.35))
        water_bonus = 20.0 if (near_water is True) else 0.0
        agri_bonus = 10.0 if (dominant_class and any(k in dominant_class.lower() for k in ["agri", "forest", "crop"])) else 0.0

        priority_score = round((area_norm * 35.0) + (ndvi_norm * 35.0) + water_bonus + agri_bonus, 1)

        c["dominant_class"] = dominant_class
        c["near_water"] = near_water
        c["water_dist_m"] = round(min_water_dist_m, 1) if min_water_dist_m < 999999.0 else None
        c["priority_score"] = priority_score

    # Return ALL clusters (every status), sorted by priority score descending
    clusters.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)
    return clusters



def get_legacy_session():
    session = requests.Session()
    adapter = LegacyRenegotiationAdapter()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

FALLBACK_LAND_USE = []
FALLBACK_MASTER_PLAN = []

@st.cache_data(ttl=86400)
def fetch_official_legend(service_name):
    url = f"https://onemapdepts.gmda.gov.in/server1/rest/services/FMDA/{service_name}/MapServer/legend?f=json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    session = get_legacy_session()
    try:
        res = session.get(url, timeout=20, verify=False, headers=headers)
        data = res.json()
        items = []
        for layer in data.get("layers", []):
            if service_name == "FMDA_Land_Use" and layer.get("layerId") != 1:
                continue
            for leg in layer.get("legend", []):
                label = leg.get("label", "").strip()
                img_b64 = leg.get("imageData", "")
                if label and img_b64:
                    items.append({
                        "label": label,
                        "image": f"data:image/png;base64,{img_b64}"
                    })
        if items:
            return items
    except Exception as e:
        print(f"FMDA Legend fetch error for {service_name}: {e}")

    # Fallback remains active if the server drops connection entirely
    return FALLBACK_LAND_USE if service_name == "FMDA_Land_Use" else FALLBACK_MASTER_PLAN

