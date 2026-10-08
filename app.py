"""
app.py — GroundTruth Enforcement Dashboard
==========================================
Streamlit front-end only.  All computation is delegated to pipeline.py.
"""

from __future__ import annotations

import io
import traceback

import folium
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

import pipeline as pl

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GroundTruth — DTP Enforcement Radar",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — tight, professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Header stripe */
    .gt-header {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 1.4rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .gt-header h1 { color: #f0f4f8; font-size: 2rem; margin: 0; }
    .gt-header p  { color: #90afc5; margin: 0.25rem 0 0; font-size: 0.95rem; }

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
    .chip-suppressed { background:#e2e3e5; color:#495057; }
    .chip-flagged    { background:#f8d7da; color:#842029; }
    .chip-ok         { background:#d1e7dd; color:#0a3622; }

    /* Alert card */
    .alert-card {
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        background: #f8f9fa;
        margin-top: 1rem;
    }
    .alert-card h3 { margin-top: 0; }
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
st.markdown(
    """
    <div class="gt-header">
        <h1>🛰️ GroundTruth — DTP Enforcement Radar</h1>
        <p>
            Faridabad District &nbsp;|&nbsp; Unauthorized Colony &amp; 
            Illegal Plot Carving Detection &nbsp;|&nbsp;
            Sentinel-2 NDVI Δ Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Run the detection pipeline (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running NDVI detection pipeline…")
def cached_detection():
    ndvi_data, clusters = pl.run_detection()
    # Convert numpy arrays to Python lists for JSON serialisation in cache
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in ndvi_data.items()}, clusters


try:
    ndvi_data_raw, clusters = cached_detection()
except Exception as exc:
    st.error(f"**Pipeline error:** {exc}")
    st.code(traceback.format_exc())
    st.stop()

# Restore numpy arrays from cache
ndvi_data: dict = {}
for k, v in ndvi_data_raw.items():
    ndvi_data[k] = np.array(v) if isinstance(v, list) else v

# ---------------------------------------------------------------------------
# TASK 5-2 — Three-column lifecycle view
# ---------------------------------------------------------------------------
st.markdown("## Parcel Lifecycle — Temporal Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-title">Baseline — Agricultural (Dec 2024)</p>', unsafe_allow_html=True)
    st.image(str(pl.T0_VISUAL), width="stretch")

with col2:
    st.markdown('<p class="section-title">Recent Pass — Suspected Clearing (May 2025)</p>', unsafe_allow_html=True)
    st.image(str(pl.T1_VISUAL), width="stretch")

with col3:
    st.markdown('<p class="section-title">Algorithmic Flag — NDVI Δ Anomaly Mask (High Risk)</p>', unsafe_allow_html=True)

    # Build a colourised NDVI-delta visualisation
    delta = ndvi_data["delta_ndvi"]
    mask  = ndvi_data["anomaly_mask"]

    # Normalise delta to [0,1] for display
    d_min, d_max = float(delta.min()), float(delta.max())
    if d_max > d_min:
        delta_norm = (delta - d_min) / (d_max - d_min)
    else:
        delta_norm = np.zeros_like(delta)

    # Compose: grey background + red overlay on masked pixels
    h, w = delta_norm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    grey = (delta_norm * 200).astype(np.uint8)
    rgba[..., 0] = grey
    rgba[..., 1] = grey
    rgba[..., 2] = grey
    rgba[..., 3] = 255
    # Red channel boost on flagged pixels
    rgba[mask == 1, 0] = np.clip(grey[mask == 1] + 180, 0, 255)
    rgba[mask == 1, 1] = np.clip(grey[mask == 1] - 60, 0, 255)
    rgba[mask == 1, 2] = np.clip(grey[mask == 1] - 60, 0, 255)

    mask_img = Image.fromarray(rgba, mode="RGBA")
    st.image(mask_img, width="stretch")

# ---------------------------------------------------------------------------
# TASK 5-3 — Pre-Filter: Artifact Suppression Log
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Pre-Filter: Artifact Suppression Log")
st.caption(
    "The pipeline labels every connected anomaly cluster and evaluates its "
    "fill-ratio (flagged pixels ÷ bounding-box area). Near-perfect rectangles "
    f"(fill_ratio ≥ {pl.FILL_RATIO_ARTIFACT_THRESHOLD}) are SUPPRESSED as sensor "
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
            "Pixels": c["pixel_count"],
            "Fill Ratio": f"{c['fill_ratio']:.4f}",
            "NDVI Δ (mean)": f"{c['ndvi_drop_mean']:.4f}",
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
    m2.metric("CANDIDATE (→ AI)", n_cand)
    m3.metric("SUPPRESSED (artifact)", n_supp)

# ---------------------------------------------------------------------------
# TASK 5-4 / 5-5 — AI Verification + Folium Map (per CANDIDATE cluster)
# ---------------------------------------------------------------------------
candidates = [c for c in clusters if c["status"] == "CANDIDATE"]

if candidates:
    st.markdown("---")
    st.markdown("## AI Verification — AWS Bedrock / Claude")

    # If more than one candidate, let the user pick
    if len(candidates) == 1:
        selected_cluster = candidates[0]
    else:
        cluster_labels = {
            f"Cluster {c['id']}  |  NDVI Δ {c['ndvi_drop_mean']:.4f}  |  ({c['lat']}, {c['lon']})": c
            for c in candidates
        }
        chosen_label = st.selectbox("Select CANDIDATE cluster to verify:", list(cluster_labels))
        selected_cluster = cluster_labels[chosen_label]

    st.markdown(
        f"**Selected cluster:** `ID {selected_cluster['id']}`  "
        f"· pixels={selected_cluster['pixel_count']}  "
        f"· fill_ratio={selected_cluster['fill_ratio']}  "
        f"· NDVI Δ={selected_cluster['ndvi_drop_mean']:.4f}  "
        f"· ({selected_cluster['lat']}°N, {selected_cluster['lon']}°E)"
    )

    # Bedrock Model & Region Configuration
    with st.expander("⚙️ Bedrock Model & AWS Region Settings", expanded=True):
        col_reg, col_model = st.columns([1, 2])
        with col_reg:
            bedrock_region = st.text_input(
                "AWS Region",
                value=pl.AWS_REGION,
                help="Region where your Bedrock model or inference profile is accessed.",
            )

        discovered = pl.list_available_inference_profiles(region=bedrock_region)
        discovered_ids = [p["id"] for p in discovered]
        
        account_profiles = [
            "au.anthropic.claude-sonnet-4-6",
            "global.anthropic.claude-sonnet-4-6",
            "au.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "apac.anthropic.claude-sonnet-4-20250514-v1:0",
            "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "au.anthropic.claude-sonnet-5",
            "global.anthropic.claude-sonnet-5",
            "global.anthropic.claude-sonnet-5-5",
            "Enter Custom Model ID / ARN",
        ]
        combined_options = discovered_ids + [k for k in account_profiles if k not in discovered_ids]

        with col_model:
            selected_choice = st.selectbox(
                "Detected / Known Profiles in Region",
                options=combined_options,
                index=0,
                help="Select an inference profile or choose 'Enter Custom Model ID / ARN'.",
            )

        default_input = "" if selected_choice == "Enter Custom Model ID / ARN" else selected_choice
        active_model_id = st.text_input(
            "Model ID or Inference Profile ID / ARN to invoke",
            value=default_input,
            placeholder="e.g. apac.anthropic.claude... or arn:aws:bedrock:...",
            help="Copy the exact Model ID or Inference Profile ARN from your AWS Bedrock Console.",
        ).strip()

        st.caption(
            "💡 **How to find your ID:** In the AWS Bedrock Console, navigate to **Cross-region inference** (or **Model catalog**), "
            "open your enabled Claude model, and copy the **Inference profile ID** or **ARN**."
        )

    dispatch_btn = st.button(
        "🚨 Dispatch Coordinates to AWS Bedrock for Visual Verification",
        type="primary",
        width="stretch",
    )

    # Initialise report state
    if "bedrock_report" not in st.session_state:
        st.session_state["bedrock_report"] = None
    if "bedrock_error" not in st.session_state:
        st.session_state["bedrock_error"] = None
    if "crop_bytes" not in st.session_state:
        st.session_state["crop_bytes"] = None

    if dispatch_btn:
        effective_model = active_model_id or selected_choice
        effective_region = bedrock_region.strip() or pl.AWS_REGION
        with st.spinner(f"Calling Claude on AWS Bedrock via {effective_model} ({effective_region})…"):
            try:
                report = pl.call_bedrock_verification(
                    cluster=selected_cluster,
                    ndvi_data=ndvi_data,
                    model_id=effective_model,
                    region=effective_region,
                )
                # Also get crop bytes for optional S3 upload
                crop_bytes = pl._crop_visual_image(
                    selected_cluster["bbox"],
                    ndvi_data["shape"],
                    pl.T1_VISUAL,
                )
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
    report: pl.InterdictionReport | None = st.session_state.get("bedrock_report")
    if report:
        is_flag = report.action == "FLAG_FOR_REVIEW"
        card_cls = "alert-high" if is_flag else "alert-low"
        action_badge = (
            "🔴 FLAG FOR REVIEW" if is_flag else "🟢 SUPPRESS — NO ACTION"
        )
        conf_pct = f"{report.confidence * 100:.0f}%"

        st.markdown(
            f"""
            <div class="alert-card {card_cls}">
                <h3>Interdiction Report — {report.parcel_id}</h3>
                <table style="width:100%; border-collapse:collapse;">
                <tr>
                    <td style="padding:4px 12px 4px 0; color:#6c757d; font-size:.85rem;">Action</td>
                    <td style="font-weight:700; font-size:1.05rem;">{action_badge}</td>
                </tr>
                <tr>
                    <td style="padding:4px 12px 4px 0; color:#6c757d; font-size:.85rem;">Violation Type</td>
                    <td>{report.violation_type}</td>
                </tr>
                <tr>
                    <td style="padding:4px 12px 4px 0; color:#6c757d; font-size:.85rem;">AI Confidence</td>
                    <td>{conf_pct}</td>
                </tr>
                <tr>
                    <td style="padding:4px 12px 4px 0; color:#6c757d; font-size:.85rem; vertical-align:top;">Reasoning</td>
                    <td style="font-style:italic;">{report.reasoning}</td>
                </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show the cropped image Claude saw
        crop = st.session_state.get("crop_bytes")
        if crop:
            with st.expander("🖼️ View image region sent to Claude"):
                st.image(crop, caption="Cropped T1 region dispatched to Bedrock")

        # Optional S3 upload button
        if crop:
            if st.button("☁️ Upload report + crop to S3", width="stretch"):
                with st.spinner("Uploading to S3…"):
                    try:
                        keys = pl.upload_to_s3(crop, report, report.parcel_id)
                        st.success(
                            f"✅ Uploaded!  "
                            f"`{keys['s3_image_key']}` & `{keys['s3_report_key']}`"
                        )
                    except Exception as exc:
                        st.error(f"S3 upload failed: {exc}")

        # ── TASK 5-5: Folium satellite map ─────────────────────────────────
        st.markdown("### 📍 Disturbance Location — Satellite View")

        lat = selected_cluster["lat"]
        lon = selected_cluster["lon"]
        ndvi_drop = selected_cluster["ndvi_drop_mean"]
        pixel_count = selected_cluster["pixel_count"]

        # Approx radius from pixel count (assuming ~10 m Sentinel-2 pixels)
        radius_m = int(((pixel_count * 100) / 3.14159) ** 0.5)   # circle area ≈ pixels × 10²

        fmap = folium.Map(
            location=[lat, lon],
            zoom_start=16,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<b>Cluster {selected_cluster['id']}</b><br>"
                f"NDVI Δ: {ndvi_drop:.4f}<br>"
                f"Pixels: {pixel_count}<br>"
                f"Fill Ratio: {selected_cluster['fill_ratio']}<br>"
                f"Status: {selected_cluster['status']}",
                max_width=220,
            ),
            tooltip="🔴 Flagged disturbance",
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

        st_folium(fmap, width="100%", height=480, returned_objects=[])

else:
    st.markdown("---")
    st.info(
        "ℹ️ No CANDIDATE clusters found. All detected anomalies were classified "
        "as sensor artefacts and suppressed. No AI dispatch required."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6c757d; font-size:0.8rem;'>"
    "GroundTruth · Hackathon Prototype · "
    "Faridabad DTP · Sentinel-2 L2A · "
    "Powered by AWS Bedrock / Claude"
    "</p>",
    unsafe_allow_html=True,
)
