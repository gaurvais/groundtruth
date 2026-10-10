import re

with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

# E.18-19: "sub-hectare" -> "~1 ha demonstrated"
text = text.replace("sub-hectare", "~1 ha demonstrated")

# Result claims
text = text.replace(
    "triggering government enforcement before permanent concrete structures are laid",
    "with the intended goal of enabling government enforcement before permanent concrete structures are laid"
)

text = text.replace(
    "This pipeline directly mitigates Urban Heat Islands",
    "This pipeline supports UHI mitigation"
)

# Photo placeholders
text = text.replace("*(Insert your ground photo here: `![Ground Photo](path/to/photo.jpg)`)*", "<!-- TODO: Insert ground photo here -->")

# TBD rows and Benchmark Text
table_old = \"\"\"| Region | Scan Window (T0 vs T1) | Raw Anomalies | After Size Filter | Suppressed | Watchlist / Candidate | Verified Outcome |
|--------|-----------------------|---------------|-------------------|------------|-----------------------|------------------|
| **Ballabgarh Agro Belt** | Aug 21, 2024 vs Sep 15, 2024 | ~204 | 45 | 32 | 13 | 1 known positive (~1 ha) flagged successfully |
| **Aravalli Ridge Corridor** | Sep 30, 2024 vs Oct 10, 2024 | ~150 | 28 | 15 | 13 | Cluster 8 properly flagged (pending seasonal) |
| **Neharpar Urban Ext.** | TBD | TBD | TBD | TBD | TBD | Pending field verification |
| **Entire Faridabad** | TBD | ~800 | 114 | 85 | 29 | Stress-test boundary |\"\"\"

table_new = \"\"\"| Region | Scan Window (T0 vs T1) | Raw Extracted | Suppressed (Artifacts) | Watchlist | Candidates | Verified Outcome |
|--------|-----------------------|---------------|-------------------|------------|-----------------------|------------------|
| **Ballabgarh Agro Belt** | Aug 21, 2024 vs Sep 15, 2024 | 45 | 32 | 13 | 0 | 1 known positive (~1 ha) flagged successfully |
| **Aravalli Ridge Corridor** | Sep 30, 2024 vs Oct 10, 2024 | 28 | 15 | 13 | 0 | Cluster 8 flagged for review; NDVI history suggests a possible crop cycle |\"\"\"

text = text.replace(table_old, table_new)

# Screenshots cleanup
text = re.sub(r"### Screenshots\\n\\* `!\[.*?\]\\(.*?\\)`\\n\\* `!\[.*?\]\\(.*?\\)`\\n\\* `!\[.*?\]\\(.*?\\)`\\n\\* `!\[.*?\]\\(.*?\\)`\\n\\* `!\[.*?\]\\(.*?\\)`", "<!-- TODO: Insert screenshots here -->", text)

# Persistence tracking line
text = text.replace(
    "Requires temporal persistence tracking to avoid flagging bare winter fields.",
    "Requires temporal persistence tracking to avoid flagging bare winter fields. Note: Persistence tracking is a planned next step."
)

# Links at the top
header_injection = \"\"\"# GroundTruth — DTP Enforcement Radar & UHI Risk Pipeline

**[Live App (TBD)]()** | **[Video Demo (TBD)]()** | **[DevPost Blog (TBD)]()**

![GroundTruth Header](Data/background_image.jpg)\"\"\"
text = text.replace("# GroundTruth — DTP Enforcement Radar & UHI Risk Pipeline\\n\\n![GroundTruth Header](Data/background_image.jpg)", header_injection)

# Deployment Readiness
ec2_notes = \"\"\"### 4. EC2 Deployment Readiness
When deploying on an Ubuntu EC2 instance:
1. System dependencies: `sudo apt-get install gdal-bin libgdal-dev` (needed for `rasterio`, `shapely`, `pyproj`).
2. IAM Role: Attach an IAM Instance Profile with Bedrock and S3 permissions. The app will automatically use the instance role (no need to hardcode keys in `.env`).
3. Swap file: Create a 2GB-4GB swap file for memory spikes during large STAC queries.
4. Run command:
   ```bash
   streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
   ```
\"\"\"
text = text.replace("---" + "\\n\\n" + "## Responsible Use", ec2_notes + "\\n---\\n\\n## Responsible Use")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(text)

print("README patched.")
