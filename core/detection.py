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
from integrations.fmda import *


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
        x, y = rio_xy(transform, centroid_row, centroid_col)
        
        # Reproject to EPSG:4326 if needed
        crs = ndvi_data["crs"]
        if crs.to_string() != "EPSG:4326":
            from rasterio.warp import transform as warp_transform
            lons, lats = warp_transform(crs, "EPSG:4326", [x], [y])
            lon, lat = lons[0], lats[0]
        else:
            lon, lat = x, y

        # Mean NDVI drop inside cluster
        ndvi_drop_mean = float(np.mean(delta[comp_mask]))

        # Artifact check
        artifact = is_artifact_shaped(comp_mask)
        is_stac = "scene_t1" in ndvi_data
        status = "SUPPRESSED" if artifact else ("WATCHLIST" if is_stac else "CANDIDATE")

        cluster_dict = {
            "id": label_id,
            "bbox": (row_min, col_min, row_max, col_max),
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "ndvi_drop_mean": round(ndvi_drop_mean, 4),
            "pixel_count": pixel_count,
            "fill_ratio": fill_ratio,
            "status": status,
        }
        
        if is_stac:
            cluster_dict.update({
                "t0_date": str(ndvi_data["t0_date"]),
                "t1_date": str(ndvi_data["t1_date"]),
                "t0_scene": ndvi_data["scene_t0"],
                "t1_scene": ndvi_data["scene_t1"],
            })
            
        clusters.append(cluster_dict)

    log.info(
        "Clusters after size filter: %d  (%d CANDIDATE, %d SUPPRESSED)",
        len(clusters),
        sum(1 for c in clusters if c["status"] == "CANDIDATE"),
        sum(1 for c in clusters if c["status"] == "SUPPRESSED"),
    )
    return clusters





def run_detection(bbox: tuple[float, float, float, float] | None = None) -> tuple[dict, list[dict]]:
    """
    Entry point used by app.py.

    Returns:
        ndvi_data  — raw arrays + transform
        clusters   — list of cluster dicts enriched with baseline land use & priority score
    """
    if USE_STAC:
        try:
            ndvi_data = stac_compute_ndvi(bbox)
        except Exception as exc:
            log.warning("STAC pipeline failed: %s. Falling back to local GeoTIFFs.", exc)
            ndvi_data = load_and_compute_ndvi()
    else:
        ndvi_data = load_and_compute_ndvi()

    clusters = extract_clusters(ndvi_data)
    clusters = enrich_clusters_with_land_use(clusters, ndvi_data)
    return ndvi_data, clusters


if __name__ == "__main__":
    data, clus = run_detection()
    for c in clus:
        print(c)
