import sys
from pathlib import Path

app_path = Path("frontend/app.py")
content = app_path.read_text(encoding="utf-8")

# Replace the failing Environmental Impact summary logic with the safe wrapped version
old_summary = """st.markdown("## Environmental Impact & UHI Risk Summary")

total_footprint_ha = ((bbox[2] - bbox[0]) * 111) * ((bbox[3] - bbox[1]) * 111) * 100
at_risk_ha = sum(c["pixel_count"] for c in clusters if c["status"] in ("CANDIDATE", "WATCHLIST")) * 0.01
water_prox_count = sum(1 for c in clusters if c.get("near_water"))

i1, i2, i3 = st.columns(3)
i1.metric("Total Area Scanned (ha)", f"{total_footprint_ha:,.0f} ha")
at_risk_label = f"{at_risk_ha:.2f} ha" if at_risk_ha > 0 else "0 ha (None Flagged)"
i2.metric("Vegetation-change area flagged (est. ha)", at_risk_label)
i3.metric("Critical Waterbody Proximity", f"{water_prox_count} Clusters")

st.info("🌱 *Cleared zones identified above are prime candidates for immediate government reclamation and plantation drives to mitigate localized UHI effects.*")"""

new_summary = """st.markdown("## Environmental Impact & UHI Risk Summary")
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
    st.warning(f"Could not load summary metrics: {e}")"""

content = content.replace(old_summary, new_summary)
app_path.write_text(content, encoding="utf-8")
print("Section 1 fixed.")
