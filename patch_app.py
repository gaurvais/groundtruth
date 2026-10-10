import sys
from pathlib import Path
import re

app_path = Path("frontend/app.py")
content = app_path.read_text(encoding="utf-8")

# 1. Update the area logic
content = content.replace(
    'total_footprint_ha = sum(c["pixel_count"] for c in clusters) * 0.01',
    '''# Calculate true scanning footprint (approximate ha)
    min_lon, min_lat, max_lon, max_lat = bbox
    total_footprint_ha = ((max_lon - min_lon) * 111) * ((max_lat - min_lat) * 111) * 100'''
)

content = content.replace(
    'at_risk_ha = sum(c["pixel_count"] for c in clusters if any(k in c.get("dominant_class", "").lower() for k in ["agri", "forest", "crop"])) * 0.01',
    'at_risk_ha = sum(c["pixel_count"] for c in clusters if c["status"] in ("CANDIDATE", "WATCHLIST")) * 0.01'
)

content = content.replace(
    'i1.metric("Total Footprint Monitored", f"{total_footprint_ha:.2f} ha")',
    'i1.metric("Total Area Scanned (ha)", f"{total_footprint_ha:,.1f} ha")'
)

content = content.replace(
    'i2.metric("Thermal Buffer Zone Lost (ha)", f"{at_risk_ha:.2f} ha")',
    'i2.metric("Vegetation-change area flagged (est. ha, pre-verification)", f"{at_risk_ha:.2f} ha" if at_risk_ha > 0 else "0 ha (All Suppressed)")'
)

# 2. Update status names in the cluster logic
content = content.replace('PROVISIONAL', 'WATCHLIST')
content = content.replace('.chip-candidate  { background:#fff3cd; color:#856404; }', '.chip-candidate  { background:#fff3cd; color:#856404; }\\n    .chip-watchlist  { background:#cce5ff; color:#004085; }')

# 3. Update Funnel Metrics
content = content.replace(
    'm2.metric("CANDIDATE (+\' AI)", n_cand)',
    '''n_watch = sum(1 for c in clusters if c["status"] == "WATCHLIST")
    m2.metric("WATCHLIST / CANDIDATE", n_cand + n_watch)'''
)

# 4. Action Queue - Update list
content = content.replace(
    'candidates = [c for c in clusters if c["status"] == "CANDIDATE"]',
    'candidates = [c for c in clusters if c["status"] in ("CANDIDATE", "WATCHLIST")]'
)

# 5. Update Action Queue columns
content = content.replace(
    '"Footprint (Pixels)": c["pixel_count"],',
    '"Area (est. ha)": f"{c[\'pixel_count\'] * 0.01:.2f}",'
)

# 6. Update the "No CANDIDATE clusters" message
no_cand_old = '''st.info(
        " No CANDIDATE clusters found. All detected anomalies were classified "
        "as sensor artefacts and suppressed. No AI dispatch required."
    )'''
no_cand_new = '''n_supp = sum(1 for c in clusters if c["status"] == "SUPPRESSED")
    n_bel = sum(1 for c in clusters if c["status"] == "BELOW_THRESHOLD")
    st.info(
        f" No CANDIDATE or WATCHLIST clusters found. \\n"
        f"Breakdown: {n_supp} suppressed as sensor artifacts, {n_bel} seasonal/below threshold."
    )'''
content = content.replace(no_cand_old, no_cand_new)

# 7. Map popup and other displays
content = content.replace(
    'f"<b>Pixels:</b> {pixel_count}<br>"',
    'f"<b>Area:</b> ~{pixel_count * 0.01:.2f} est. ha<br>"'
)

# 8. Replace st.bar_chart for Land Use with Altair
old_bar = 'st.bar_chart(pd.Series(lu_counts), use_container_width=True)'
new_bar = '''import altair as alt
    chart_df = pd.DataFrame(list(lu_counts.items()), columns=["Land Use", "Area (est. ha)"])
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Area (est. ha):Q", title="Area (ha)"),
        y=alt.Y("Land Use:N", sort="-x", title="", axis=alt.Axis(labelLimit=300)),
        color=alt.value("#4c78a8"),
        tooltip=["Land Use", "Area (est. ha)"]
    ).properties(height=350).configure_axis(labelFontSize=12, titleFontSize=14)
    st.altair_chart(chart, use_container_width=True)'''
content = content.replace(old_bar, new_bar)

# 9. Timeseries Chart - Adding units and caution
old_ts = '''st.line_chart(
            ts_df.set_index("Date")["Median NDVI"],
            use_container_width=True,
        )'''
new_ts = '''import altair as alt
        ts_chart = alt.Chart(ts_df).mark_line(point=True).encode(
            x=alt.X("Date:T", title="Acquisition Date"),
            y=alt.Y("Median NDVI:Q", title="NDVI (10m Resolution)"),
            tooltip=["Date:T", "Median NDVI:Q"]
        ).properties(height=300).configure_axis(labelFontSize=12, titleFontSize=14)
        st.altair_chart(ts_chart, use_container_width=True)
        st.caption(" Note: If NDVI strongly recovers later in the year, the initial drop was likely a seasonal harvest rather than permanent concrete carving.")'''
content = content.replace(old_ts, new_ts)

# 10. Update "forensic visual inspector" to "visual triage assistant"
content = content.replace('forensic visual inspector', 'visual triage assistant')

app_path.write_text(content, encoding="utf-8")
print("App.py patched.")
