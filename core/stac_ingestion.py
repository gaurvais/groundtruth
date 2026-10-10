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


CACHE_DIR = DATA_DIR / "cache"


def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    """Short stable hash of a bounding box for cache keys."""
    return hashlib.md5(f"{bbox}".encode()).hexdigest()[:10]


def _mgrs_tile_id(item) -> str:
    """Extract MGRS tile identifier from a STAC item's properties."""
    props = item.properties
    # Try grid:code first (e.g. "MGRS-43RGM"), fall back to mgrs:* fields
    gc = props.get("grid:code", "")
    if gc:
        return gc.replace("MGRS-", "")
    zone = props.get("mgrs:utm_zone", "")
    band = props.get("mgrs:latitude_band", "")
    sq = props.get("mgrs:grid_square", "")
    if zone and band and sq:
        return f"{zone}{band}{sq}"
    return ""


def _scl_valid_fraction(item, bbox_lonlat: tuple[float, float, float, float]) -> float:
    """Read only the SCL window and return fraction of valid pixels inside bbox."""
    scl_href = item.assets["scl"].href
    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    with rasterio.Env(**env_opts):
        with rasterio.open(scl_href) as src:
            # Transform bbox from EPSG:4326 → dataset CRS
            dst_bounds = transform_bounds("EPSG:4326", src.crs, *bbox_lonlat)
            win = from_bounds(*dst_bounds, transform=src.transform)
            # Clamp window to dataset extent
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            scl = src.read(1, window=win)
    if scl.size == 0:
        return 0.0
    invalid = np.isin(scl, list(SCL_INVALID_CLASSES))
    return float(1.0 - invalid.sum() / scl.size)


def pick_scenes(
    bbox_lonlat: tuple[float, float, float, float],
    t1_target_date: dt.date | str,
    force_dates: tuple[dt.date | str, dt.date | str] | None = None,
) -> dict:
    """
    Discover a T0/T1 scene pair from Sentinel-2 L2A via STAC.

    Parameters
    ----------
    bbox_lonlat : (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    t1_target_date : target date for the post-disturbance pass
    force_dates : optional (t0_date, t1_date) — bypasses gap rule for validation

    Returns
    -------
    dict with keys:
        t1_item, t0_item, t1_valid_frac, t0_valid_frac,
        t1_date, t0_date, mgrs_tile
    Raises RuntimeError with a clear reason if no qualifying pair is found.
    """
    if not HAS_PYSTAC:
        raise RuntimeError(
            "pystac-client is not installed. Run: pip install pystac-client"
        )

    if isinstance(t1_target_date, str):
        t1_target_date = dt.date.fromisoformat(t1_target_date)

    catalog = STACClient.open(STAC_API_URL)
    aoi_geom = box(*bbox_lonlat)

    # ----- force_dates mode: find specific items -----
    if force_dates is not None:
        d0 = dt.date.fromisoformat(str(force_dates[0]))
        d1 = dt.date.fromisoformat(str(force_dates[1]))
        t1_item, t1_vf = _find_item_on_date(catalog, bbox_lonlat, aoi_geom, d1)
        t0_item, t0_vf = _find_item_on_date(catalog, bbox_lonlat, aoi_geom, d0)
        return {
            "t1_item": t1_item, "t0_item": t0_item,
            "t1_valid_frac": t1_vf, "t0_valid_frac": t0_vf,
            "t1_date": d1, "t0_date": d0,
            "mgrs_tile": _mgrs_tile_id(t1_item),
        }

    # ----- normal mode -----
    search_start = t1_target_date - dt.timedelta(days=STAC_SEARCH_DAYS_BACK)
    search = catalog.search(
        collections=[STAC_COLLECTION],
        bbox=bbox_lonlat,
        datetime=f"{search_start.isoformat()}/{t1_target_date.isoformat()}",
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=80,
    )
    items = list(search.items())
    log.info("STAC returned %d items in search window.", len(items))

    # Filter: geometry must CONTAIN the AOI
    covering = [
        it for it in items
        if shape(it.geometry).contains(aoi_geom)
    ]
    log.info("Items whose geometry fully contains AOI: %d", len(covering))
    if not covering:
        raise RuntimeError(
            f"No Sentinel-2 items fully cover bbox {bbox_lonlat} in the last "
            f"{STAC_SEARCH_DAYS_BACK} days before {t1_target_date}."
        )

    # Pick T1 = newest with valid_fraction >= threshold
    t1_item = None
    t1_vf = 0.0
    for it in covering:
        vf = _scl_valid_fraction(it, bbox_lonlat)
        log.info("  %s  valid=%.2f%%", it.id, vf * 100)
        if vf >= MIN_VALID_FRACTION:
            t1_item = it
            t1_vf = vf
            break  # already sorted newest first
    if t1_item is None:
        raise RuntimeError(
            f"No scene with valid_fraction >= {MIN_VALID_FRACTION:.0%} found "
            f"in the last {STAC_SEARCH_DAYS_BACK} days."
        )

    t1_date = t1_item.datetime.date()
    t1_tile = _mgrs_tile_id(t1_item)
    log.info("T1 selected: %s  date=%s  tile=%s  valid=%.1f%%",
             t1_item.id, t1_date, t1_tile, t1_vf * 100)

    # Pick T0 = same MGRS tile, 5–30 days before T1, best valid fraction
    #   tie-break: closest to 10 days gap
    t0_candidates: list[tuple] = []
    for it in covering:
        if _mgrs_tile_id(it) != t1_tile:
            continue
        it_date = it.datetime.date()
        gap = (t1_date - it_date).days
        if T0_GAP_DAYS_MIN <= gap <= T0_GAP_DAYS_MAX:
            vf = _scl_valid_fraction(it, bbox_lonlat)
            if vf >= MIN_VALID_FRACTION:
                t0_candidates.append((it, it_date, vf, gap))

    if not t0_candidates:
        raise RuntimeError(
            f"No valid T0 scene (same tile {t1_tile}, {T0_GAP_DAYS_MIN}–"
            f"{T0_GAP_DAYS_MAX} days before {t1_date}, "
            f"valid >= {MIN_VALID_FRACTION:.0%})."
        )

    # Sort: best valid fraction first; tie-break by distance to ideal gap
    t0_candidates.sort(key=lambda x: (-x[2], abs(x[3] - T0_GAP_IDEAL_DAYS)))
    t0_item, t0_date, t0_vf, gap = t0_candidates[0]
    log.info("T0 selected: %s  date=%s  gap=%dd  valid=%.1f%%",
             t0_item.id, t0_date, gap, t0_vf * 100)

    return {
        "t1_item": t1_item, "t0_item": t0_item,
        "t1_valid_frac": t1_vf, "t0_valid_frac": t0_vf,
        "t1_date": t1_date, "t0_date": t0_date,
        "mgrs_tile": t1_tile,
    }



def cached_pick_scenes(bbox_lonlat, t1_target_date=None, force_dates=None):
    import json
    import hashlib
    from pystac import Item
    
    if t1_target_date is None:
        t1_target_date = dt.date.today()
        
    # Serialize args for hash
    def _str(d): return d.isoformat() if isinstance(d, dt.date) else str(d)
    fd_str = f"{_str(force_dates[0])}_{_str(force_dates[1])}" if force_dates else "None"
    s = f"{bbox_lonlat}_{_str(t1_target_date)}_{fd_str}"
    cache_key = hashlib.md5(s.encode()).hexdigest()
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"pick_scenes_{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            log.info(f"Loaded STAC metadata from JSON cache: {cache_file.name}")
            return {
                "t1_item": Item.from_dict(data["t1_item"]),
                "t0_item": Item.from_dict(data["t0_item"]),
                "t1_valid_frac": data["t1_valid_frac"],
                "t0_valid_frac": data["t0_valid_frac"],
                "t1_date": dt.date.fromisoformat(data["t1_date"]),
                "t0_date": dt.date.fromisoformat(data["t0_date"]),
                "mgrs_tile": data["mgrs_tile"],
            }
        except Exception as e:
            log.warning(f"Failed to load pick_scenes cache: {e}")
            
    res = pick_scenes(bbox_lonlat, t1_target_date, force_dates)
    
    try:
        data = {
            "t1_item": res["t1_item"].to_dict(),
            "t0_item": res["t0_item"].to_dict(),
            "t1_valid_frac": res["t1_valid_frac"],
            "t0_valid_frac": res["t0_valid_frac"],
            "t1_date": res["t1_date"].isoformat(),
            "t0_date": res["t0_date"].isoformat(),
            "mgrs_tile": res["mgrs_tile"],
        }
        with open(cache_file, "w") as f:
            json.dump(data, f)
        log.info(f"Saved STAC metadata to JSON cache: {cache_file.name}")
    except Exception as e:
        log.warning(f"Failed to save pick_scenes cache: {e}")
        
    return res

def _find_item_on_date(catalog, bbox_lonlat, aoi_geom, target_date: dt.date):
    """Helper: find the best item on a specific date ± 1 day."""
    d0 = target_date - dt.timedelta(days=1)
    d1 = target_date + dt.timedelta(days=1)
    search = catalog.search(
        collections=[STAC_COLLECTION],
        bbox=bbox_lonlat,
        datetime=f"{d0.isoformat()}/{d1.isoformat()}",
        max_items=20,
    )
    for it in search.items():
        if not shape(it.geometry).contains(aoi_geom):
            continue
        vf = _scl_valid_fraction(it, bbox_lonlat)
        if vf >= MIN_VALID_FRACTION:
            return it, vf
    raise RuntimeError(
        f"No valid Sentinel-2 scene found on or near {target_date} for bbox."
    )


def _get_reflectance_params(asset) -> tuple[float, float]:
    """Extract scale/offset from asset's raster:bands metadata, with fallback."""
    try:
        rb = asset.extra_fields["raster:bands"][0]
        scale = rb.get("scale", DEFAULT_REFLECTANCE_SCALE)
        offset = rb.get("offset", DEFAULT_REFLECTANCE_OFFSET)
        return float(scale), float(offset)
    except (KeyError, IndexError, TypeError):
        return DEFAULT_REFLECTANCE_SCALE, DEFAULT_REFLECTANCE_OFFSET


def read_bands(
    item, bbox_lonlat: tuple[float, float, float, float],
) -> dict:
    """
    Windowed read of red, nir, scl from a STAC item over the given bbox.

    Returns dict:
        red (float32), nir (float32), valid_mask (bool),
        window_transform, crs, scene_id, datetime,
        red_window (Window object for reuse)
    """
    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    red_href = item.assets["red"].href
    nir_href = item.assets["nir"].href
    scl_href = item.assets["scl"].href

    red_scale, red_offset = _get_reflectance_params(item.assets["red"])
    nir_scale, nir_offset = _get_reflectance_params(item.assets["nir"])

    with rasterio.Env(**env_opts):
        # -- Red (10 m) ---------------------------------------------------
        with rasterio.open(red_href) as red_src:
            dst_bounds = transform_bounds("EPSG:4326", red_src.crs, *bbox_lonlat)
            red_win = from_bounds(*dst_bounds, transform=red_src.transform)
            # Snap to integer pixel offsets
            red_win = red_win.round_offsets().round_lengths()
            red_dn = red_src.read(1, window=red_win).astype(np.float32)
            win_transform = red_src.window_transform(red_win)
            crs = red_src.crs
            red_shape = red_dn.shape

        # -- NIR (10 m) — reuse same window --------------------------------
        with rasterio.open(nir_href) as nir_src:
            nir_dn = nir_src.read(1, window=red_win).astype(np.float32)

        # -- SCL (20 m) — resample to red grid -----------------------------
        with rasterio.open(scl_href) as scl_src:
            scl_bounds = transform_bounds("EPSG:4326", scl_src.crs, *bbox_lonlat)
            scl_win = from_bounds(*scl_bounds, transform=scl_src.transform)
            scl_win = scl_win.round_offsets().round_lengths()
            scl = scl_src.read(
                1, window=scl_win,
                out_shape=red_shape,
                resampling=Resampling.nearest,
            )

    # DN == 0 is nodata in both red and nir
    nodata_mask = (red_dn == 0) | (nir_dn == 0)

    # Apply reflectance: DN * scale + offset, then clip to [0, 1]
    red = np.clip(red_dn * red_scale + red_offset, 0.0, 1.0)
    nir = np.clip(nir_dn * nir_scale + nir_offset, 0.0, 1.0)

    # SCL valid mask
    scl_invalid = np.isin(scl, list(SCL_INVALID_CLASSES))

    # Combined valid mask (True = valid pixel)
    valid_mask = ~nodata_mask & ~scl_invalid

    # Dilate invalid mask by 1 pixel to exclude boundary artifacts
    invalid_dilated = ndimage.binary_dilation(~valid_mask, iterations=1)
    valid_mask = ~invalid_dilated

    return {
        "red": red,
        "nir": nir,
        "valid_mask": valid_mask,
        "window_transform": win_transform,
        "crs": crs,
        "scene_id": item.id,
        "datetime": item.datetime,
        "red_window": red_win,
        "shape": red_shape,
    }


def _cache_path(mgrs_tile: str, t0_date: dt.date, t1_date: dt.date,
                bbox: tuple[float, float, float, float]) -> Path:
    """Build deterministic .npz cache path."""
    bh = _bbox_hash(bbox)
    return CACHE_DIR / f"{mgrs_tile}_{t0_date}_{t1_date}_{bh}.npz"


def stac_compute_ndvi(
    bbox_lonlat: tuple[float, float, float, float] | None = None,
    t1_target_date: dt.date | str | None = None,
    force_dates: tuple | None = None,
) -> dict:
    """
    Full STAC-based NDVI pipeline: discover scenes, read bands, compute
    reflectance-calibrated NDVI and delta.

    Returns the same dict schema as load_and_compute_ndvi() so the downstream
    clustering / enrichment code can run unchanged:
        ndvi_t0, ndvi_t1, delta_ndvi, anomaly_mask,
        transform, crs, shape,
        (plus extra STAC metadata: scene_t0, scene_t1, t0_date, t1_date,
         mgrs_tile, valid_t0, valid_t1)
    """
    bbox = bbox_lonlat or DEFAULT_BBOX_LONLAT
    target = t1_target_date or dt.date.today()
    if isinstance(target, str):
        target = dt.date.fromisoformat(target)

    # --- Try cache first ---
    if force_dates:
        d0 = dt.date.fromisoformat(str(force_dates[0]))
        d1 = dt.date.fromisoformat(str(force_dates[1]))
        cp = _cache_path("auto", d0, d1, bbox)
    else:
        cp = None  # we don't know dates yet

    if cp and cp.exists():
        log.info("Loading STAC NDVI from cache: %s", cp)
        return _load_stac_cache(cp)

    # --- Discover scenes ---
    scenes = cached_pick_scenes(bbox, target, force_dates=force_dates)
    mgrs_tile = scenes["mgrs_tile"]
    cp = _cache_path(mgrs_tile, scenes["t0_date"], scenes["t1_date"], bbox)

    if cp.exists():
        log.info("Loading STAC NDVI from cache: %s", cp)
        return _load_stac_cache(cp)

    # --- Read bands ---
    log.info("Reading T1 bands from COG: %s ...", scenes["t1_item"].id)
    t1_data = read_bands(scenes["t1_item"], bbox)
    log.info("Reading T0 bands from COG: %s ...", scenes["t0_item"].id)
    t0_data = read_bands(scenes["t0_item"], bbox)

    # --- NDVI ---
    ndvi_t0 = _safe_ndvi(t0_data["red"], t0_data["nir"])
    ndvi_t1 = _safe_ndvi(t1_data["red"], t1_data["nir"])

    # Delta only where both dates are valid, else NaN
    both_valid = t0_data["valid_mask"] & t1_data["valid_mask"]
    delta_ndvi = np.where(both_valid, ndvi_t0 - ndvi_t1, np.nan)

    # Anomaly mask (NaN stays excluded — becomes False in comparison)
    anomaly_mask = np.where(
        np.isnan(delta_ndvi), 0, (delta_ndvi > NDVI_THRESHOLD)
    ).astype(np.uint8)

    result = {
        "ndvi_t0": ndvi_t0,
        "ndvi_t1": ndvi_t1,
        "delta_ndvi": delta_ndvi.astype(np.float32),
        "anomaly_mask": anomaly_mask,
        "transform": t1_data["window_transform"],
        "crs": t1_data["crs"],
        "shape": t1_data["shape"],
        # Extra STAC metadata
        "scene_t0": t0_data["scene_id"],
        "scene_t1": t1_data["scene_id"],
        "t0_date": scenes["t0_date"],
        "t1_date": scenes["t1_date"],
        "mgrs_tile": mgrs_tile,
        "valid_t0": t0_data["valid_mask"],
        "valid_t1": t1_data["valid_mask"],
    }

    # --- Cache ---
    _save_stac_cache(cp, result)
    return result


def _safe_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI with safe division; zero-sum → NaN."""
    np.seterr(divide="ignore", invalid="ignore")
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return ndvi.astype(np.float32)


def _save_stac_cache(path: Path, data: dict) -> None:
    """Persist windowed NDVI arrays + metadata to .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "scene_t0": data.get("scene_t0", ""),
        "scene_t1": data.get("scene_t1", ""),
        "t0_date": str(data.get("t0_date", "")),
        "t1_date": str(data.get("t1_date", "")),
        "mgrs_tile": data.get("mgrs_tile", ""),
        "crs": str(data["crs"]),
        "transform": np.array(list(data["transform"])[:6]),
    }
    np.savez_compressed(
        path,
        ndvi_t0=data["ndvi_t0"],
        ndvi_t1=data["ndvi_t1"],
        delta_ndvi=data["delta_ndvi"],
        anomaly_mask=data["anomaly_mask"],
        valid_t0=data.get("valid_t0", np.ones_like(data["anomaly_mask"], dtype=bool)),
        valid_t1=data.get("valid_t1", np.ones_like(data["anomaly_mask"], dtype=bool)),
        **meta,
    )
    log.info("Cached STAC NDVI to %s", path)


def _load_stac_cache(path: Path) -> dict:
    """Reload windowed NDVI arrays from .npz cache."""
    npz = np.load(path, allow_pickle=False)
    tf_vals = npz["transform"]
    transform = rasterio.transform.Affine(*tf_vals)
    crs_str = str(npz["crs"])
    crs = rasterio.crs.CRS.from_user_input(crs_str)
    t0_d = str(npz["t0_date"])
    t1_d = str(npz["t1_date"])
    return {
        "ndvi_t0": npz["ndvi_t0"],
        "ndvi_t1": npz["ndvi_t1"],
        "delta_ndvi": npz["delta_ndvi"],
        "anomaly_mask": npz["anomaly_mask"],
        "transform": transform,
        "crs": crs,
        "shape": npz["ndvi_t0"].shape,
        "scene_t0": str(npz.get("scene_t0", "")),
        "scene_t1": str(npz.get("scene_t1", "")),
        "t0_date": dt.date.fromisoformat(t0_d) if t0_d else None,
        "t1_date": dt.date.fromisoformat(t1_d) if t1_d else None,
        "mgrs_tile": str(npz.get("mgrs_tile", "")),
        "valid_t0": npz.get("valid_t0"),
        "valid_t1": npz.get("valid_t1"),
    }



def cluster_timeseries(
    cluster: dict,
    ndvi_data: dict,
    months: int = 12,
) -> list:
    """
    Walk backwards over the last `months` calendar months. For each month,
    find the clearest Sentinel-2 L2A scene that fully contains the cluster bbox,
    do a windowed read of red+nir over the cluster bounding box, and return the
    median NDVI for cloud-free pixels.

    Results are cached to data/cache/ts_<cluster_id>_<bbox_hash>.json.

    Returns
    -------
    list of {"date": "YYYY-MM-DD", "ndvi": float}, sorted oldest-first.
    Empty list if STAC is unavailable.
    """
    if not HAS_PYSTAC:
        return []

    row_min, col_min, row_max, col_max = cluster["bbox"]
    tf = ndvi_data["transform"]
    crs = ndvi_data["crs"]
    from rasterio.transform import xy as _rio_xy
    x0, y0 = _rio_xy(tf, row_min, col_min, offset="ul")
    x1, y1 = _rio_xy(tf, row_max, col_max, offset="lr")
    lons, lats = warp_transform(crs, "EPSG:4326", [x0, x1], [y0, y1])
    cluster_bbox = (min(lons), min(lats), max(lons), max(lats))

    bh = _bbox_hash(cluster_bbox)
    cache_file = CACHE_DIR / f"ts_{cluster['id']}_{bh}.json"
    if cache_file.exists():
        import json as _json
        return _json.loads(cache_file.read_text())

    catalog = STACClient.open(STAC_API_URL)
    aoi_geom = box(*cluster_bbox)
    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    import calendar, json as _json
    end_date = dt.date.today()
    results = []

    for m in range(months - 1, -1, -1):
        year = end_date.year
        month = end_date.month - m
        while month <= 0:
            month += 12
            year -= 1
        month_start = dt.date(year, month, 1)
        month_end = dt.date(year, month, calendar.monthrange(year, month)[1])
        try:
            search = catalog.search(
                collections=[STAC_COLLECTION],
                bbox=list(cluster_bbox),
                datetime=f"{month_start.isoformat()}/{month_end.isoformat()}",
                max_items=20,
            )
            items = [it for it in search.items() if shape(it.geometry).contains(aoi_geom)]
            if not items:
                continue
            best_item, best_vf = None, 0.0
            for it in items:
                vf = _scl_valid_fraction(it, cluster_bbox)
                if vf > best_vf:
                    best_vf, best_item = vf, it
            if best_item is None or best_vf < 0.5:
                continue

            red_href = best_item.assets["red"].href
            nir_href = best_item.assets["nir"].href
            scl_href = best_item.assets["scl"].href
            red_scale, red_offset = _get_reflectance_params(best_item.assets["red"])
            nir_scale, nir_offset = _get_reflectance_params(best_item.assets["nir"])

            with rasterio.Env(**env_opts):
                with rasterio.open(red_href) as red_src:
                    dst_b = transform_bounds("EPSG:4326", red_src.crs, *cluster_bbox)
                    red_win = from_bounds(*dst_b, transform=red_src.transform).round_offsets().round_lengths()
                    red_dn = red_src.read(1, window=red_win).astype(np.float32)
                    rshape = red_dn.shape
                with rasterio.open(nir_href) as nir_src:
                    nir_dn = nir_src.read(1, window=red_win).astype(np.float32)
                with rasterio.open(scl_href) as scl_src:
                    scl_b = transform_bounds("EPSG:4326", scl_src.crs, *cluster_bbox)
                    scl_win = from_bounds(*scl_b, transform=scl_src.transform).round_offsets().round_lengths()
                    scl = scl_src.read(1, window=scl_win, out_shape=rshape, resampling=Resampling.nearest)

            valid = (red_dn > 0) & (nir_dn > 0) & ~np.isin(scl, list(SCL_INVALID_CLASSES))
            if valid.sum() < 10:
                continue
            red = np.clip(red_dn * red_scale + red_offset, 0.0, 1.0)
            nir = np.clip(nir_dn * nir_scale + nir_offset, 0.0, 1.0)
            denom = nir + red
            ndvi_arr = np.where((denom > 0) & valid, (nir - red) / denom, np.nan)
            median_ndvi = float(np.nanmedian(ndvi_arr))
            scene_date = best_item.datetime.date() if best_item.datetime else month_start
            results.append({"date": str(scene_date), "ndvi": round(median_ndvi, 4)})
        except Exception:
            continue

    results.sort(key=lambda r: r["date"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(_json.dumps(results))
    return results
