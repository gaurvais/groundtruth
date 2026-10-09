# GroundTruth: DTP Enforcement Radar 🛰️

Early-warning enforcement dashboard for the **District Town Planner (DTP) in Faridabad**.

GroundTruth detects unauthorized colonies and illegal plot carving on agricultural land **before** permanent concrete structures are constructed. By fusing multi-spectral satellite analysis (Sentinel-2 NDVI drop) with automated artifact suppression and multimodal LLM verification (Claude on AWS Bedrock), GroundTruth provides an auditable, cost-efficient, and rapid interdiction workflow.

---

## 🌟 The Problem & Solution

1. **The Challenge:** Enforcing land-use regulations after permanent buildings or boundary walls are built leads to expensive litigation, demolition resistance, and irreversible loss of arable land.
2. **Early Detection:** Illegal colonies begin with subtle activities: vegetation clearing, dirt road carving, and boundary demarcation.
3. **Low-Cost, High-Precision Pipeline:**
   - **Multi-Spectral Anomaly Detection:** Compares baseline farmland ($T_0$) with recent passes ($T_1$) using Sentinel-2 B04 (Red) and B08 (NIR) bands to calculate vegetation drop ($\Delta\text{NDVI} > 0.15$).
   - **Algorithmic Artifact Suppression:** Labels connected components and filters sensor swath edges / nodata artifacts using bounding box fill-ratio heuristics ($\ge 0.58$), saving unnecessary AI inference costs.
   - **Multimodal AI Verification:** Dispatches genuine candidate coordinate crops to **Claude Sonnet on AWS Bedrock** to visually identify road grids, plot boundaries, and soil compaction.
   - **Enforcement Dashboard:** Visualizes parcel lifecycles, artifact suppression logs, structured interdiction reports, and interactive satellite maps via Folium.

---

## 🏗️ Architecture

```
Sentinel-2 Imagery (T0: Baseline, T1: Recent)
                     │
                     ▼
          [ NDVI Delta Calculation ]
            NDVI = (NIR - Red) / (NIR + Red)
            Flag if (NDVI_t0 - NDVI_t1) > 0.15
                     │
                     ▼
       [ Connected Component Clustering ]
           scipy.ndimage.label + Bounding Box
                     │
                     ▼
          [ Artifact Filter (Fill-Ratio) ]
          ┌──────────┴──────────┐
          ▼                     ▼
     SUPPRESSED             CANDIDATE
 (Sensor / Swath)        (Genuine Anomaly)
  (Zero AI Cost)                │
                                ▼
                    [ AWS Bedrock / Claude ]
                  Multimodal Vision Analysis
                                │
                                ▼
                    [ Pydantic Report ]
                 InterdictionReport (JSON)
                                │
                                ▼
                     [ Streamlit UI & Maps ]
                    Interactive Satellite View
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- AWS Account with Amazon Bedrock access to Anthropic Claude models (e.g., `au.anthropic.claude-sonnet-4-6` or `apac.anthropic.claude-3-5-sonnet-20241022-v2:0`)

### 2. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/GroundTruth.git
cd GroundTruth
pip install -r requirements.txt
```

### 3. Configure AWS Credentials

Ensure your AWS credentials and region are set:

```powershell
# Windows PowerShell
$env:AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
$env:AWS_DEFAULT_REGION = "ap-southeast-2"
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
GroundTruth/
├── app.py              # Streamlit enforcement radar UI & Folium maps
├── pipeline.py         # Geospatial NDVI processing, artifact filter & Bedrock client
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusion rules
├── README.md           # Documentation
└── Data/               # Satellite data
    ├── t0_b04.tif.tiff # T0 Red band (Dec 2024)
    ├── t0_b08.tif.tiff # T0 NIR band (Dec 2024)
    ├── t0_visual.png   # T0 True-color composite
    ├── t1_b04.tif.tiff # T1 Red band (May 2025)
    ├── t1_b08.tif.tiff # T1 NIR band (May 2025)
    └── t1_visual.png   # T1 True-color composite
```

---

## 🚀 Recent Updates

- **NumPy Soft-Alpha Master Plan Overlay:** Applied a dynamic server-side Python (NumPy) alpha mask to instantly remove the solid white background on the FMDA ArcGIS Master Plan exports while preserving the tinted statutory grid overlay, seamlessly integrating with Folium.
- **Live FMDA REST Legend Integration:** Fetches canonical cartographic symbology rules directly from the state GIS MapServer `/legend?f=json` endpoints. Utilizes a custom `LegacyRenegotiationAdapter` (on `requests.Session`) to bypass outdated OpenSSL protocol errors when connecting to the government server.
- **Deterministic AI Verification:** Calibrated the AWS Bedrock/Claude prompt to focus strictly on physical ground alterations rather than making definitive legal judgments. Set inference configuration to `temperature: 0.0, maxTokens: 600` for highly deterministic, repeatable outputs.
- **Environmental Impact Panel:** Included a high-level Streamlit metric dashboard that aggregates pixel cluster counts, converting footprints to hectares and flagging intersection violations with protected agricultural land and 100m waterbody buffers.

---

## 📄 Output Schema: Interdiction Report

When coordinates are dispatched to AWS Bedrock, Claude returns structured JSON validated against the `InterdictionReport` Pydantic model:

```json
{
  "parcel_id": "FARIDABAD-001",
  "confidence": 0.92,
  "violation_type": "Unauthorized Plot Carving / Road Grid",
  "action": "FLAG_FOR_REVIEW",
  "reasoning": "Linear soil compaction patterns and rectangular plot subdivisions visible across former agricultural parcel."
}
```
