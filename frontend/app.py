"""
app.py — GroundTruth Enforcement Dashboard
==========================================
Streamlit front-end only.  All computation is delegated to pipeline.py.
"""

import io
import traceback
import base64

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

import sys
from pathlib import Path
# Add project root to sys.path so modules (utils, core, integrations) can be resolved
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.config import list_available_inference_profiles, FILL_RATIO_ARTIFACT_THRESHOLD, BEDROCK_MODEL_ID, AWS_REGION, LegacyRenegotiationAdapter, T1_VISUAL, USE_STAC
from core.detection import stac_compute_ndvi, extract_clusters, load_and_compute_ndvi
from core.stac_ingestion import cluster_timeseries
from integrations.fmda import enrich_clusters_with_land_use

from core.bedrock import call_bedrock_verification, upload_to_s3, _crop_visual_image, _crop_stac_visual, InterdictionReport
from integrations.fmda import fetch_transparent_master_plan, fetch_official_legend

import base64

def set_background(image_path: str):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        
        css = f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(14, 17, 23, 0.45), rgba(14, 17, 23, 0.45)),
                        url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        pass



# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GroundTruth — DTP Enforcement Radar",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — tight, professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Header removed */

    /* Metric chips */
    .chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .chip-candidate  { background:#fff3cd; color:#856404; }
    .chip-watchlist  { background:#cce5ff; color:#004085; }
    .chip-suppressed { background:#e2e3e5; color:#495057; }
    .chip-flagged    { background:#f8d7da; color:#842029; }
    .chip-ok         { background:#d1e7dd; color:#0a3622; }

    /* Alert card */
    .alert-card {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        background: #1E2330;
        color: #FFFFFF !important;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .alert-card h3 { 
        margin-top: 0; 
        color: #FFFFFF !important; 
        font-weight: 700;
        font-size: 1.25rem;
    }
    .alert-card td {
        color: #FFFFFF !important;
    }
    .alert-high { border-left: 6px solid #dc3545; }
    .alert-low  { border-left: 6px solid #198754; }

    /* Section titles */
    .section-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #6c757d;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stImage"] img { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("GroundTruth — DTP Enforcement Radar")
st.markdown("Faridabad District | Unauthorized Colony & Illegal Plot Carving Detection | Sentinel-2 NDVI Δ Analysis")

# ---------------------------------------------------------------------------
# Sidebar & State
# ---------------------------------------------------------------------------
st.sidebar.header("Scan Parameters")

REGIONS = {
    "Aravalli Ridge Corridor": (77.16, 28.28, 77.30, 28.46),
    "Ballabgarh Agro Belt": (77.28, 28.22, 77.42, 28.34),
    "Neharpar Urban Extension": (77.32, 28.32, 77.44, 28.45),
    "Entire Faridabad District": (77.15, 28.20, 77.55, 28.52),
}

region_name = st.sidebar.selectbox("Select Target Region", list(REGIONS.keys()))
timeframe_name = st.sidebar.selectbox("Select Timeframe", [
    "Live Rapid Surveillance (Latest 5-14 Days)",
    "Historical Ground-Truth Audit (Sept 2024 Enforcement)"
])

if timeframe_name == "Historical Ground-Truth Audit (Sept 2024 Enforcement)":
    force_dates = ("2024-08-21", "2024-09-15")
else:
    force_dates = None

run_scan = st.sidebar.button("Run Satellite Scan")

if run_scan:
    st.session_state.scan_run = True
    st.session_state.bbox = REGIONS[region_name]
    st.session_state.force_dates = force_dates

if not st.session_state.get("scan_run", False):
    set_background("Data/background_image.jpg")
    st.markdown("### Automated Satellite Surveillance & Urban Heat Island (UHI) Risk Pipeline")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### AWS Open Data Registry (S3)\nStreams Sentinel-2 Cloud-Optimized GeoTIFFs directly from Amazon S3 to detect early-stage vegetation loss—the leading indicator of thermal buffer destruction.")
    with col2:
        st.markdown("### Morphological Pruning\nCross-references FMDA Master Plan zoning to isolate rapid concrete plot carving that accelerates urban heat.")
    with col3:
        st.markdown("### Amazon Bedrock (Claude)\nKeeps spatial telemetry securely within the AWS ecosystem, utilizing multimodal LLMs to halt encroachment before permanent thermal islands form.")
        
    st.divider()
    st.markdown("*Built for the WeMakeDevs / AWS Hackathon*")
    st.stop()

# ---------------------------------------------------------------------------
# Run the detection pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_detection(bbox, force_dates):
    from scipy import ndimage as _ndi
    try:
        ndvi_data_res = stac_compute_ndvi(bbox_lonlat=bbox, force_dates=force_dates)
    except Exception as exc:
        print(f"STAC failed: {exc}")
        ndvi_data_res = load_and_compute_ndvi()

    # --- Funnel counts (before any filter) ---
    _, n_raw = _ndi.label(ndvi_data_res["anomaly_mask"])
    c = extract_clusters(ndvi_data_res)
    c = enrich_clusters_with_land_use(c, ndvi_data_res)

    funnel = {
        "raw_components":        int(n_raw),
        "after_min_area":        len(c),
        "after_artifact_filter": sum(1 for x in c if x["status"] != "SUPPRESSED"),
        "suppressed":            sum(1 for x in c if x["status"] == "SUPPRESSED"),
        "sent_to_bedrock":       0,
        "flag_for_review":       0,
        "needs_field_verification": 0,
        "suppress_no_action":    0,
    }

    return (
        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in ndvi_data_res.items()},
        c,
        funnel,
    )

try:
    with st.spinner("Querying AWS Earth Search STAC & Processing NDVI..."):
        ndvi_data_raw, clusters, funnel = cached_detection(
            st.session_state.bbox, st.session_state.force_dates
        )
except Exception as exc:
    st.error(f"**Pipeline error:** {exc}")
    st.code(traceback.format_exc())
    st.stop()

# Restore numpy arrays from cache
ndvi_data: dict = {}
for k, v in ndvi_data_raw.items():
    ndvi_data[k] = np.array(v) if isinstance(v, list) else v


# ---------------------------------------------------------------------------
# Scan Summary
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Scan Summary")

t0_date = ndvi_data.get("t0_date", "local")
t1_date = ndvi_data.get("t1_date", "local")
total_ha = sum(c["pixel_count"] * 0.01 for c in clusters)
near_water_count = sum(1 for c in clusters if c.get("near_water"))

# 1 — Detection funnel
st.markdown("### Detection Funnel")

action = ""
has_report = False
if st.session_state.get("bedrock_report"):
    rpt = st.session_state["bedrock_report"]
    action = getattr(rpt, "action", "")
    has_report = True

fc1, fc2, fc3, fc4, fc5 = st.columns(5)
fc1.metric("Raw Anomalies", funnel["raw_components"])
fc2.metric("After Size Filter", funnel["after_min_area"])
fc3.metric("Artifacts Suppressed", funnel["suppressed"])
fc4.metric("Sent to AI", 1 if has_report else 0)
fc5.metric("Flagged for Action", 1 if action == "FLAG_FOR_REVIEW" else 0)

with st.expander("Fill-ratio & Status per cluster"):
    for c in clusters:
        st.write(f"Cluster {c['id']:>4} | fill_ratio={c['fill_ratio']:.4f} | status={c['status']}")

# 2 — Vegetation-loss area by land-use class
st.markdown("### Vegetation-loss area (proxy) by Baseline Land Use")
mc1, mc2 = st.columns(2)
mc1.metric("Total flagged area (ha)", f"{total_ha:.2f}")
mc2.metric("Clusters within 100 m of water", near_water_count)

lu_ha: dict[str, float] = {}
for c in clusters:
    lu = c.get("dominant_class") or "Unknown"
    lu_ha[lu] = round(lu_ha.get(lu, 0.0) + c["pixel_count"] * 0.01, 3)

if lu_ha:
    import altair as alt
    lu_df = pd.DataFrame({"Land Use": list(lu_ha.keys()), "Area (ha)": list(lu_ha.values())})
    chart = alt.Chart(lu_df).mark_bar().encode(
        x=alt.X("Area (ha):Q", title="Area (ha)"),
        y=alt.Y("Land Use:N", sort="-x", title="", axis=alt.Axis(labelLimit=300)),
        color=alt.value("#4c78a8"),
        tooltip=["Land Use", "Area (ha)"]
    ).properties(height=350).configure_axis(labelFontSize=12, titleFontSize=14)
    st.altair_chart(chart, width='stretch')

# 4 — NDVI timeseries for best cluster (highest priority)
st.markdown("### NDVI Timeseries — Top Cluster (12 months)")
if clusters:
    # Prefer CANDIDATE / PROVISIONAL — never plot a SUPPRESSED cluster as the headline
    active = [c for c in clusters if c["status"] in ("CANDIDATE", "PROVISIONAL")]
    top_cluster = max(
        active if active else clusters,
        key=lambda c: c.get("priority_score", 0.0),
    )

    @st.cache_data(show_spinner=False, ttl=86400)
    def _cached_timeseries(cluster_id, bbox, force_dates_key):
        # find the cluster dict again (cluster list is not hashable)
        cl = next((c for c in clusters if c["id"] == cluster_id), None)
        if cl is None:
            return []
        return cluster_timeseries(cl, ndvi_data, months=12)

    with st.spinner("Fetching NDVI timeseries from STAC (cached after first run)..."):
        try:
            ts = _cached_timeseries(
                top_cluster["id"],
                st.session_state.bbox,
                str(st.session_state.force_dates),
            )
        except Exception:
            ts = []

    if ts:
        import altair as alt
        ts_df = pd.DataFrame(ts)
        ts_chart = alt.Chart(ts_df).mark_line(point=True).encode(
            x=alt.X("date:T", title="Acquisition Date"),
            y=alt.Y("ndvi:Q", title="NDVI (10m Resolution)"),
            tooltip=["date:T", "ndvi:Q"]
        ).properties(height=300).configure_axis(labelFontSize=12, titleFontSize=14)
        st.altair_chart(ts_chart, width='stretch')
        st.caption(f"Cluster {top_cluster['id']} (Lat: {top_cluster['lat']}, Lon: {top_cluster['lon']}) — Harvest dips then regrows; sustained low NDVI is consistent with clearing. *Note: If NDVI strongly recovers later in the year, the initial drop was likely a seasonal harvest.*")
    else:
        st.info("Timeseries unavailable (STAC not reachable or no clear scenes found).")

# ---------------------------------------------------------------------------
# TASK 5-3 — Pre-Filter: Artifact Suppression Log
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Pre-Filter: Artifact Suppression Log")
st.caption(
    "The pipeline labels every connected anomaly cluster and evaluates its "
    "fill-ratio (flagged pixels ÷ bounding-box area). Near-perfect rectangles "
    f"(fill_ratio ≥ {FILL_RATIO_ARTIFACT_THRESHOLD}) are SUPPRESSED as sensor "
    "artefacts without invoking the AI, saving cost and avoiding false positives."
)

if not clusters:
    st.info("No clusters found above the minimum pixel threshold.")
else:
    import pandas as pd

    df = pd.DataFrame([
        {
            "Cluster ID": c["id"],
            "Status": c["status"],
            "Priority Score": c.get("priority_score", 0.0),
            "Baseline Land Use (FMDA)": c.get("dominant_class", "Agriculture Cropland"),
            "Near Water (<100m)": " Yes" if c.get("near_water") else "No",
            "Pixels": c["pixel_count"],
            "Fill Ratio": f"{c['fill_ratio']:.4f}",
            "NDVI Drop (Δ)": f"{c['ndvi_drop_mean']:.4f}",
            "Centroid Lat": c["lat"],
            "Centroid Lon": c["lon"],
        }
        for c in clusters
    ])

    def _style_status(val):
        if val == "CANDIDATE":
            return "background-color:#fff3cd; color:#856404; font-weight:600;"
        if val == "SUPPRESSED":
            return "background-color:#e2e3e5; color:#495057; font-weight:600;"
        return ""

    styled_df = df.style.map(_style_status, subset=["Status"])
    st.dataframe(styled_df, width="stretch", hide_index=True)

    n_cand = sum(1 for c in clusters if c["status"] == "CANDIDATE")
    n_supp = sum(1 for c in clusters if c["status"] == "SUPPRESSED")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Clusters", len(clusters))
    n_watch = sum(1 for c in clusters if c["status"] == "WATCHLIST")
    m2.metric("WATCHLIST / CANDIDATE", n_cand + n_watch)
    m3.metric("SUPPRESSED (artifact)", n_supp)

# ---------------------------------------------------------------------------
# Government Action Queue (Ranked Priority & FMDA Baseline Land Use)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Environmental Impact & UHI Risk Summary")
try:
    from utils.config import calculate_scanned_ha
    bbox = st.session_state.bbox
    total_footprint_ha = calculate_scanned_ha(bbox)
    
    cand_ha = sum(c["pixel_count"] for c in clusters if c["status"] == "CANDIDATE") * 0.01
    watch_ha = sum(c["pixel_count"] for c in clusters if c["status"] == "WATCHLIST") * 0.01
    water_prox_count = sum(1 for c in clusters if c.get("near_water"))

    i1, i2, i3 = st.columns(3)
    i1.metric("Area scanned (est. ha)", f"{total_footprint_ha:,.0f} ha")
    i2.metric("Candidate (persistent)", f"{cand_ha:.2f} est. ha")
    i3.metric("Watchlist (new, unconfirmed)", f"{watch_ha:.2f} est. ha")

    st.caption("**Status Guide:** SUPPRESSED (geometric artifact), BELOW_THRESHOLD (seasonal/crop), WATCHLIST (new change, not yet confirmed), CANDIDATE (persistent). *Note: Persistence tracking is a planned next step. Until implemented, anomalies will remain WATCHLIST.*")
    
except Exception as e:
    st.warning(f"Could not load summary metrics: {e}")

candidates = [c for c in clusters if c["status"] in ("CANDIDATE", "WATCHLIST")]

if candidates:
    st.markdown("---")
    st.markdown("##  Government Action Queue (Ranked Interdiction Priority)")
    st.caption(
        "Interdiction candidates ranked by enforcement urgency. Scoring fuses Sentinel-2 vegetation clearing drop "
        "(NDVI Δ), disturbance footprint, official **FMDA baseline land use** (Level1_des), and **100m waterbody buffer** proximity."
    )

    action_queue_data = [
        {
            "Rank": idx,
            "Priority Score": c.get("priority_score", 0.0),
            "Cluster ID": c["id"],
            "Baseline Land Use (FMDA survey)": c.get("dominant_class", "Agriculture Cropland"),
            "Near Waterbody (<100m)": " YES" if c.get("near_water") else "No",
            "NDVI Drop (Δ)": f"{c['ndvi_drop_mean']:.4f}",
            "Area (est. ha)": f"{c['pixel_count'] * 0.01:.2f}",
            "Fill Ratio": f"{c['fill_ratio']:.4f}",
            "Coordinates": f"{c['lat']}°N, {c['lon']}°E",
        }
        for idx, c in enumerate(candidates, 1)
    ]
    df_queue = pd.DataFrame(action_queue_data)

    def _style_queue(row):
        styles = [""] * len(row)
        score = row["Priority Score"]
        if score >= 50:
            styles[1] = "background-color:#fee2e2; color:#991b1b; font-weight:700;"
        elif score >= 30:
            styles[1] = "background-color:#fef3c7; color:#92400e; font-weight:700;"
        else:
            styles[1] = "background-color:#e0f2fe; color:#075985; font-weight:600;"
        if row["Near Waterbody (<100m)"] == " YES":
            styles[4] = "background-color:#fee2e2; color:#991b1b; font-weight:700;"
        return styles

    styled_queue = df_queue.style.apply(_style_queue, axis=1)
    st.dataframe(styled_queue, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("## AI Verification — AWS Bedrock / Claude")

    # If more than one candidate, let the user pick
    if len(candidates) == 1:
        selected_cluster = candidates[0]
    else:
        cluster_labels = {
            f"Rank #{idx} | Cluster {c['id']} | Priority: {c.get('priority_score', 0)} | {c.get('dominant_class', 'Agriculture')} | NDVI Δ {c['ndvi_drop_mean']:.4f}": c
            for idx, c in enumerate(candidates, 1)
        }
        chosen_label = st.selectbox("Select CANDIDATE cluster to verify:", list(cluster_labels))
        selected_cluster = cluster_labels[chosen_label]

    st.markdown(
        f"**Selected Cluster:** `ID {selected_cluster['id']}` &nbsp;|&nbsp; "
        f"**Priority Score:** `{selected_cluster.get('priority_score', 'N/A')}` &nbsp;|&nbsp; "
        f"**Baseline Land Use (FMDA):** `{selected_cluster.get('dominant_class', 'Agriculture Cropland')}` &nbsp;|&nbsp; "
        f"**Near Waterbody (<100m):** `{' Yes' if selected_cluster.get('near_water') else 'No'}` &nbsp;|&nbsp; "
        f"**NDVI Δ:** `{selected_cluster['ndvi_drop_mean']:.4f}` &nbsp;|&nbsp; "
        f"**Pixels:** `{selected_cluster['pixel_count']}`"
    )

    dispatch_btn = st.button(
        "Dispatch Coordinates to AWS Bedrock for Visual Verification",
        type="primary",
        width='stretch',
    )

    # Initialise report state
    if "bedrock_report" not in st.session_state:
        st.session_state["bedrock_report"] = None
    if "bedrock_error" not in st.session_state:
        st.session_state["bedrock_error"] = None
    if "crop_bytes" not in st.session_state:
        st.session_state["crop_bytes"] = None

    if dispatch_btn:
        with st.spinner(f"Calling Claude on AWS Bedrock via {BEDROCK_MODEL_ID} ({AWS_REGION})…"):
            try:
                report = call_bedrock_verification(
                    cluster=selected_cluster,
                    ndvi_data=ndvi_data,
                    model_id=BEDROCK_MODEL_ID,
                    region=AWS_REGION,
                )
                if "scene_t1" in ndvi_data:
                    crop_bytes = _crop_stac_visual(selected_cluster, ndvi_data, "scene_t1")
                else:
                    crop_bytes = _crop_visual_image(selected_cluster["bbox"], ndvi_data["shape"], T1_VISUAL)
                st.session_state["bedrock_report"] = report
                st.session_state["bedrock_error"] = None
                st.session_state["crop_bytes"] = crop_bytes
            except Exception as exc:
                st.session_state["bedrock_report"] = None
                st.session_state["bedrock_error"] = str(exc)
                st.session_state["_exc_trace"] = traceback.format_exc()

    # ── Display error ──────────────────────────────────────────────────────
    if st.session_state.get("bedrock_error"):
        st.error(f"**Bedrock call failed:** {st.session_state['bedrock_error']}")
        with st.expander("Stack trace"):
            st.code(st.session_state.get("_exc_trace", ""))

    # ── Display structured alert card ─────────────────────────────────────
    report: InterdictionReport | None = st.session_state.get("bedrock_report")
    if report:
        is_flag = report.action == "FLAG_FOR_REVIEW"
        card_cls = "alert-high" if is_flag else "alert-low"
        action_badge = (
            " FLAG FOR REVIEW" if is_flag else " SUPPRESS — NO ACTION"
        )
        conf_pct = f"{report.confidence * 100:.0f}%"

        badge_color = "#b91c1c" if is_flag else "#15803d"
        badge_bg = "#fee2e2" if is_flag else "#dcfce7"

        land_use_text = selected_cluster.get('dominant_class', 'Agriculture Cropland')
        water_dist_val = selected_cluster.get('water_dist_m', 'N/A')
        water_prox_text = (
            " Within 100m Buffer Zone"
            if selected_cluster.get("near_water")
            else f"Outside 100m Buffer ({water_dist_val} m)"
        )

        st.markdown(
            f"""
            <div class="alert-card {card_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid #e5e7eb; padding-bottom:0.75rem;">
                    <h3 style="margin:0; color:#111827 !important; font-size:1.3rem; font-weight:700;">Interdiction Report — {report.parcel_id}</h3>
                    <span style="background:{badge_bg}; color:{badge_color} !important; font-weight:700; font-size:0.9rem; padding:4px 12px; border-radius:999px; border:1px solid {badge_color}40;">
                        {action_badge}
                    </span>
                </div>
                <table style="width:100%; border-collapse:collapse; color:#1f2937 !important;">
                <tr>
                    <td style="padding:6px 16px 6px 0; color:#4b5563 !important; font-size:0.9rem; font-weight:600; width:170px;">Violation Type</td>
                    <td style="color:#111827 !important; font-size:0.95rem; font-weight:600;">{report.violation_type}</td>
                </tr>
                <tr>
                    <td style="padding:6px 16px 6px 0; color:#4b5563 !important; font-size:0.9rem; font-weight:600;">Baseline Land Use (FMDA)</td>
                    <td style="color:#111827 !important; font-size:0.95rem; font-weight:600;">{land_use_text}</td>
                </tr>
                <tr>
                    <td style="padding:6px 16px 6px 0; color:#4b5563 !important; font-size:0.9rem; font-weight:600;">Waterbody Proximity</td>
                    <td style="color:#111827 !important; font-size:0.95rem; font-weight:600;">{water_prox_text}</td>
                </tr>
                <tr>
                    <td style="padding:6px 16px 6px 0; color:#4b5563 !important; font-size:0.9rem; font-weight:600;">AI Confidence</td>
                    <td style="color:#111827 !important; font-size:0.95rem; font-weight:600;">{conf_pct}</td>
                </tr>
                <tr>
                    <td style="padding:6px 16px 6px 0; color:#4b5563 !important; font-size:0.9rem; font-weight:600; vertical-align:top;">Reasoning</td>
                    <td style="color:#1f2937 !important; font-size:0.95rem; line-height:1.55; font-style:italic;">{report.reasoning}</td>
                </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show the cropped image Claude saw
        crop = st.session_state.get("crop_bytes")
        if crop:
            with st.expander(" View raw Bedrock payload & image crop"):
                st.image(crop, caption="Cropped T1 (Post-Disturbance) dispatched to Bedrock")
                st.markdown(f"**Model ID:** `{report.model_used}`  |  **Region:** `{report.region_used}`")
                st.code(report.raw_json, language="json")

        # Optional S3 upload button
        if crop:
            if st.button(" Upload report + crop to S3", width="stretch"):
                with st.spinner("Uploading to S3…"):
                    try:
                        keys = upload_to_s3(crop, report, report.parcel_id)
                        st.success(
                            f" Uploaded!  "
                            f"`{keys['s3_image_key']}` & `{keys['s3_report_key']}`"
                        )
                    except Exception as exc:
                        st.error(f"S3 upload failed: {exc}")

    # ── TASK 5-5: Folium satellite map + Master Plan 2031 ──────────────
    st.markdown("### Disturbance Location — Satellite & Master Plan View")

    st.info("Use the Layer Control icon  in the top right of the map to toggle official FMDA cartography overlays.")

    lat = selected_cluster["lat"]
    lon = selected_cluster["lon"]
    ndvi_drop = selected_cluster["ndvi_drop_mean"]
    pixel_count = selected_cluster["pixel_count"]
    dominant_lu = selected_cluster.get("dominant_class", "Agriculture Cropland")
    near_water_flag = selected_cluster.get("near_water", False)
    priority_sc = selected_cluster.get("priority_score", 0.0)

    # Approx radius from pixel count (assuming ~10 m Sentinel-2 pixels)
    radius_m = int(((pixel_count * 100) / 3.14159) ** 0.5)

    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=16,
        tiles=None,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name=" Satellite Basemap (Esri)",
    ).add_to(fmap)

    min_lon, min_lat, max_lon, max_lat = 77.275, 28.305, 77.297, 28.327

    # 1. FMDA Master Plan 2031 Overlay (Server-side alpha masked)
    mp_path = fetch_transparent_master_plan(bbox=(min_lon, min_lat, max_lon, max_lat))
    import base64
    with open(mp_path, "rb") as image_file:
        mp_b64 = base64.b64encode(image_file.read()).decode("utf-8")
    mp_data_uri = f"data:image/png;base64,{mp_b64}"

    folium.raster_layers.ImageOverlay(
        image=mp_data_uri,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.75,
        transparent=True,
        name=" FMDA Master Plan 2031 (Statutory Zoning)",
        overlay=True,
        control=True,
        show=False,
    ).add_to(fmap)

    # 2. FMDA Land Use Overlay (Official Patterns)
    lu_export_url = f"https://onemapdepts.gmda.gov.in/server1/rest/services/FMDA/FMDA_Land_Use/MapServer/export?bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=4326&size=1400,1400&layers=show:1&format=png32&transparent=true&f=image"
    folium.raster_layers.ImageOverlay(
        image=lu_export_url,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=1.0,
        transparent=True,
        name=" FMDA Baseline Land Use (Mock/Cached)",
        overlay=True,
        control=True,
        show=False,
    ).add_to(fmap)

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b>Cluster {selected_cluster['id']}</b><br>"
            f"<b>Priority Score:</b> {priority_sc}<br>"
            f"<b>Baseline Land Use (FMDA):</b> {dominant_lu}<br>"
            f"<b>Near Water:</b> {' Yes (<100m)' if near_water_flag else 'No'}<br>"
            f"<b>NDVI Δ:</b> {ndvi_drop:.4f}<br>"
            f"<b>Area:</b> ~{pixel_count * 0.01:.2f} est. ha<br>"
            f"<b>Fill Ratio:</b> {selected_cluster['fill_ratio']}<br>"
            f"<b>Status:</b> {selected_cluster['status']}",
            max_width=260,
        ),
        tooltip=f" Flagged disturbance (Score: {priority_sc})",
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
    ).add_to(fmap)

    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color="#dc3545",
        fill=True,
        fill_color="#dc3545",
        fill_opacity=0.20,
        tooltip=f"Approx. disturbed area (~{radius_m} m radius)",
    ).add_to(fmap)

    folium.LayerControl(position="topright", collapsed=True).add_to(fmap)

    st_folium(fmap, width="100%", height=480, returned_objects=[])

    with st.expander(" Official FMDA Statutory Legend (Direct from State GIS Server)", expanded=False):
        tab1, tab2 = st.tabs([" Land Use Cadastral Survey", " Master Plan 2031 Zoning"])

        with tab1:
            lu_items = fetch_official_legend("FMDA_Land_Use")
            if lu_items:
                cols = st.columns(3)
                for idx, item in enumerate(lu_items):
                    with cols[idx % 3]:
                        st.markdown(
                            f'<div style="display:flex; align-items:center; margin-bottom:8px;">'
                            f'<img src="{item["image"]}" style="margin-right:10px; border:1px solid #ccc; width:20px; height:20px;" />'
                            f'<span style="font-size:12px;">{item["label"]}</span>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
            else:
                st.caption("Mock FMDA Land Use legend unavailable.")

        with tab2:
            mp_items = fetch_official_legend("FMDA_MasterPlan2031")
            if mp_items:
                cols = st.columns(3)
                for idx, item in enumerate(mp_items):
                    with cols[idx % 3]:
                        st.markdown(
                            f'<div style="display:flex; align-items:center; margin-bottom:8px;">'
                            f'<img src="{item["image"]}" style="margin-right:10px; border:1px solid #ccc; width:20px; height:20px;" />'
                            f'<span style="font-size:12px;">{item["label"]}</span>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
            else:
                st.caption("Official FMDA Master Plan legend service unavailable.")

else:
    st.markdown("---")
    n_supp = sum(1 for c in clusters if c["status"] == "SUPPRESSED")
    n_bel = sum(1 for c in clusters if c["status"] == "BELOW_THRESHOLD")
    st.info(
        f" No CANDIDATE or WATCHLIST clusters found.\\n\\n"
        f"**Breakdown:** {n_supp} suppressed as sensor artifacts, {n_bel} seasonal/below threshold."
    )


