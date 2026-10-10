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
    candidates = [c for c in clusters if c["status"] in ("CANDIDATE", "PROVISIONAL")]
    candidates.sort(key=lambda x: x["priority_score"], reverse=True)
    suppressed = [c for c in clusters if c["status"] == "SUPPRESSED"]

    return candidates + suppressed



import streamlit as st
from utils.config import LegacyRenegotiationAdapter
import requests
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

