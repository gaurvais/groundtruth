import sys
from pathlib import Path
import json
import datetime as dt

sys.path.insert(0, str(Path.cwd()))
from core.detection import run_detection
from core.stac_ingestion import pick_scenes

REGIONS = {
    "Ballabgarh Agro Belt": (77.28, 28.22, 77.42, 28.34),
    "Aravalli Ridge Corridor": (77.16, 28.28, 77.30, 28.46)
}

benchmark_dates = {
    "Ballabgarh Agro Belt": dt.date(2024, 9, 20),
    "Aravalli Ridge Corridor": dt.date(2024, 10, 20)
}

results = []
for region_name, bbox in REGIONS.items():
    print(f"Scanning {region_name}...")
    try:
        force_date = benchmark_dates.get(region_name)
        scenes = pick_scenes(bbox, force_date)
        t0 = scenes["t0_date"]
        t1 = scenes["t1_date"]
        
        ndvi_data, clusters = run_detection(bbox, force_dates=force_date)
        raw_count = len(clusters)
        
        suppressed = sum(1 for c in clusters if c["status"] == "SUPPRESSED")
        watchlist = sum(1 for c in clusters if c["status"] == "WATCHLIST")
        candidates = sum(1 for c in clusters if c["status"] == "CANDIDATE")
        
        results.append({
            "Region": region_name,
            "Scan Window": f"{t0} vs {t1}",
            "Raw Extracted": raw_count,
            "Suppressed": suppressed,
            "Watchlist": watchlist,
            "Candidates": candidates,
            "Scan Date": t1
        })
    except Exception as e:
        print(f"Failed {region_name}: {e}")

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Benchmarks generated: benchmark_results.json")
