# GroundTruth — DTP Enforcement Radar & UHI Risk Pipeline

![GroundTruth Header](Data/background_image.jpg)

**GroundTruth** is an enterprise-grade satellite surveillance dashboard built for the District Town Planner (DTP) of Faridabad. It acts as an automated, cloud-native radar to detect unauthorized colonies and illegal plot carving on agricultural lands. 

By detecting early-stage vegetation loss, the pipeline not only enforces statutory zoning but also identifies the destruction of critical thermal buffers—serving as a frontline defense against **Urban Heat Island (UHI)** formation and light pollution. It fuses AWS Open Data, geospatial algorithms, and Multimodal Generative AI (Amazon Bedrock) into a highly actionable GovTech dashboard.

*Built for the WeMakeDevs / AWS Hackathon.*

---

## 🚀 Key Features

1. **Cloud-Native STAC Ingestion (AWS Open Data):** Streams 10m-resolution Sentinel-2 L2A Cloud-Optimized GeoTIFFs (COGs) directly from Amazon S3 via the STAC API. Completely bypasses massive local downloads using targeted windowed reads.
2. **Morphological Pruning & Clustering:** Uses `scipy.ndimage` to compute NDVI drops, extract contiguous clusters, and calculate geometric fill-ratios to automatically suppress harvest noise and sensor artifacts.
3. **FMDA GIS Cross-Referencing:** Spatially joins anomaly clusters with Faridabad Master Plan 2031 and official Baseline Land Use cadastral surveys (e.g., "Agriculture Cropland", waterbody proximity).
4. **Multimodal AI Verification (Amazon Bedrock):** Dispatches high-probability anomaly crops to Amazon Bedrock (Claude). The LLM acts as a forensic visual inspector, verifying anthropogenic concrete carving vs. natural drying, assigning a 0–100% confidence score, and evaluating UHI risks.
5. **Urban Climate Resilience (UHI):** Translates "Agricultural Land At Risk" into "Thermal Buffer Zone Lost", creating a direct action queue for government reclamation and plantation drives.
6. **Professional GovTech UI:** A sleek, high-contrast, dark-themed Streamlit dashboard featuring detection funnels, ranked action queues, and interactive Folium cartography.

---

## 🏗️ Architecture & Folder Structure

The project has been heavily refactored for maintainability and modularity:

```text
GroundTruth/
├── .streamlit/
│   └── config.toml           # AWS dark theme configurations
├── core/
│   ├── bedrock.py            # AWS Bedrock/Claude invocation, AI prompts, and S3 uploads
│   ├── detection.py          # NDVI calculation, clustering, and morphological pruning
│   └── stac_ingestion.py     # pystac-client search, COG window reads, and timeseries caching
├── frontend/
│   └── app.py                # Main Streamlit dashboard UI
├── integrations/
│   └── fmda.py               # Spatially joining clusters with FMDA Land Use / Master Plan
├── utils/
│   └── config.py             # Constants, threshold variables, region bounding boxes
├── Data/
│   ├── background_image.jpg  # Hero background image
│   └── cache/                # STAC queries cache (.npz, .json) for instant reloads
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment variables
└── README.md                 # Project documentation
```

---

## ⚙️ Technology Stack & Dependencies

*   **Frontend / UI:** [Streamlit](https://streamlit.io/), `folium`, `streamlit-folium`, `pandas`
*   **Geospatial & Ingestion:** `pystac-client`, `rasterio`, `shapely`, `pyproj`
*   **Algorithms & Math:** `numpy`, `scipy`
*   **Cloud & AI (AWS):** `boto3` (Amazon Bedrock Runtime, Amazon S3)

---

## 🛠️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/GroundTruth.git
cd GroundTruth
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory (or copy `.env.example`) and configure your AWS credentials to enable Amazon Bedrock and S3 uploads:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-west-2
# Ensure your IAM role has access to Bedrock models (e.g., anthropic.claude-3-5-sonnet-20241022-v2:0)
```

*(Note: STAC API reads from the AWS Earth Search registry are anonymous and do not require signed requests).*

---

## 💻 Usage Instructions

### Running the Dashboard
Launch the Streamlit application using the following command from the root directory:

```bash
streamlit run frontend/app.py
```

### Dashboard Workflow
1. **Landing Page:** You will be greeted by the AWS Cloud architecture overview.
2. **Scan Parameters (Sidebar):** Select your target region (e.g., "Ballabgarh Agro Belt") and a timeframe (Live Rapid Surveillance or Historical Audit).
3. **Run Satellite Scan:** Click the primary button. The pipeline will:
   * Query the STAC API for the best cloud-free T0 and T1 Sentinel-2 scenes.
   * Compute the NDVI anomaly mask natively on the cloud.
   * Filter, prune, and cross-reference the data with FMDA GIS layers.
4. **Action Queue:** Review the "Environmental Impact & UHI Risk Summary". High-priority candidates will be listed in a ranked dataframe.
5. **AI Verification:** Select a candidate cluster and click **"Dispatch Coordinates to AWS Bedrock"**. Claude will analyze the visual crop and provide an official Interdiction Report (FLAG_FOR_REVIEW or SUPPRESS).
6. **Folium Map:** Explore the interactive map at the bottom to view the exact cluster bounds overlaid on the statutory Master Plan 2031 zoning layer.

---

## 🤝 License & Acknowledgements

Developed for the **WeMakeDevs / AWS Hackathon**.
Satellite Imagery provided by ESA / Copernicus via AWS Open Data (Earth Search by Element 84).
Master Plan / GIS layers utilized for hackathon demonstration purposes (mock integrations inspired by FMDA).
