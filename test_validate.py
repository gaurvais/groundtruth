import sys
from pathlib import Path
import datetime as dt
import numpy as np
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from core.stac_ingestion import pick_scenes, read_bands, _scl_valid_fraction, STAC_COLLECTION
from utils.config import MIN_CLUSTER_PIXELS, FILL_RATIO_ARTIFACT_THRESHOLD, NDVI_THRESHOLD
from pystac_client import Client

REGIONS = {
    "Aravalli Ridge Corridor": (77.16, 28.28, 77.30, 28.46),
    "Ballabgarh Agro Belt": (77.28, 28.22, 77.42, 28.34),
}

PERSISTENCE_MIN_SCENES = 3
PERSISTENCE_MIN_DAYS = 30
MIN_VALID_FRACTION = 0.85

def compute_ndvi_safe(red, nir, valid_mask):
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))
    ndvi[~valid_mask] = np.nan
    return ndvi

def fetch_history(bbox, t1_date):
    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    d_start = t1_date - dt.timedelta(days=400)
    d_end = t1_date
    search = catalog.search(
        collections=[STAC_COLLECTION],
        bbox=bbox,
        datetime=f"{d_start.isoformat()}/{d_end.isoformat()}",
        max_items=100
    )
    items = list(search.items())
    
    baseline_date = t1_date - dt.timedelta(days=365)
    baseline_candidates = [it for it in items if abs((it.datetime.date() - baseline_date).days) <= 45]
    
    recent_items = []
    for it in items:
        if _scl_valid_fraction(it, bbox) >= MIN_VALID_FRACTION:
            recent_items.append(it)
            if len(recent_items) >= 5:
                break
                
    baseline_items = []
    for it in baseline_candidates:
        if _scl_valid_fraction(it, bbox) >= MIN_VALID_FRACTION:
            baseline_items.append(it)
            if len(baseline_items) >= 3:
                break
                
    return recent_items, baseline_items

def test_region(region_name):
    print(f"\\n--- Testing {region_name} ---")
    bbox = REGIONS[region_name]
    scenes = pick_scenes(bbox, t1_target_date=dt.date.today())
    t1_date = scenes["t1_date"]
    
    recent_items, baseline_items = fetch_history(bbox, t1_date)
    if not recent_items or not baseline_items:
        print("Not enough history.")
        return
        
    t1_item = recent_items[0]
    t0_item = baseline_items[0]
    
    b_t0 = read_bands(t0_item, bbox)
    ndvi_t0 = compute_ndvi_safe(b_t0["red"], b_t0["nir"], b_t0["valid_mask"])
    del b_t0
    
    b_t1 = read_bands(t1_item, bbox)
    ndvi_t1 = compute_ndvi_safe(b_t1["red"], b_t1["nir"], b_t1["valid_mask"])
    del b_t1
    
    delta_ndvi = np.nan_to_num(ndvi_t0, nan=0) - np.nan_to_num(ndvi_t1, nan=0)
    anomaly_mask = (delta_ndvi > NDVI_THRESHOLD).astype(np.uint8)
    
    labeled_array, n_labels = ndimage.label(anomaly_mask)
    indices = np.arange(1, n_labels + 1)
    pixel_counts = ndimage.sum(np.ones_like(anomaly_mask), labeled_array, indices)
    
    valid_mask = (pixel_counts >= MIN_CLUSTER_PIXELS)
    valid_indices = indices[valid_mask]
    
    recent_ndvis = []
    recent_dates = []
    for it in recent_items[:4]:
        b = read_bands(it, bbox)
        n = compute_ndvi_safe(b["red"], b["nir"], b["valid_mask"])
        recent_ndvis.append(n)
        recent_dates.append(it.datetime.date())
        del b
        
    base_ndvis = []
    for it in baseline_items:
        b = read_bands(it, bbox)
        n = compute_ndvi_safe(b["red"], b["nir"], b["valid_mask"])
        base_ndvis.append(n)
        del b
        
    baseline_median = np.nanmedian(np.stack(base_ndvis), axis=0)
    
    top_indices = valid_indices[np.argsort(pixel_counts[valid_mask])[::-1]][:3]
    for idx in top_indices:
        print(f"\\nCluster ID {idx} (Pixels: {pixel_counts[idx-1]} / {pixel_counts[idx-1]*0.01:.2f} ha)")
        
        comp_mask = (labeled_array == idx)
        b_mean = np.nanmean(baseline_median[comp_mask])
        print(f"Baseline mean NDVI (median of last year): {b_mean:.3f}")
        
        rec_means = []
        for i, n in enumerate(recent_ndvis):
            r_mean = np.nanmean(n[comp_mask])
            rec_means.append(r_mean)
            print(f"Recent {recent_dates[i]} NDVI: {r_mean:.3f}")
            
        persistent = False
        span_days = 0
        consecutive_count = 0
        for i in range(len(rec_means)):
            if rec_means[i] < (b_mean - 0.15) or rec_means[i] < 0.25:
                consecutive_count += 1
                span = (recent_dates[0] - recent_dates[i]).days
                if consecutive_count >= PERSISTENCE_MIN_SCENES and span >= PERSISTENCE_MIN_DAYS:
                    persistent = True
                    span_days = span
                    break
            else:
                consecutive_count = 0
                
        print(f"Persistent? {persistent} (Span: {span_days} days)")

if __name__ == "__main__":
    test_region("Aravalli Ridge Corridor")
    test_region("Ballabgarh Agro Belt")
