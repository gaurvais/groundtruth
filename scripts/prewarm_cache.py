import sys
from pathlib import Path

# Ensure cache dir exists
Path("Data/cache").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path.cwd()))
from core.detection import run_detection
from frontend.app import REGIONS

print("Pre-warming STAC cache for all preset regions...")
for region_name, bbox in REGIONS.items():
    print(f"Warming {region_name}...")
    try:
        run_detection(bbox)
        print(f"[OK] {region_name}")
    except Exception as e:
        print(f"[FAILED] {region_name}: {e}")

print("Pre-warming complete!")
