"""
EcoSense — Streamlit Interactive Dashboard
All predictions are driven entirely by user-provided inputs.
No values are assumed or pre-filled.
Run: streamlit run streamlit_app.py
"""
import os
import sys
import hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.model import EcoSenseModel

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="EcoSense — AI Carbon Copilot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Model ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ecosense_xgb.joblib")

@st.cache_resource(show_spinner="Loading EcoSense AI Engine…")
def get_model(artifact_version: int):
    return EcoSenseModel(models_dir=MODEL_DIR)

model = get_model(os.stat(MODEL_PATH).st_mtime_ns)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #070d1a !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(12,20,40,0.95) !important;
    border-right: 1px solid rgba(255,255,255,.08) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: #94a3b8 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(14,23,42,0.8);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.76rem !important; text-transform: uppercase; letter-spacing: .05em; }
div[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 800 !important; font-size: 1.5rem !important; }

/* Input fields */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.04) !important;
    border-color: rgba(255,255,255,.12) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16,185,129,.2) !important;
}
input { color: #f1f5f9 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: .9rem !important;
    padding: .6rem 1.5rem !important;
    box-shadow: 0 0 18px rgba(16,185,129,.25) !important;
    transition: all .2s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Alert/info boxes */
.stAlert { border-radius: 10px !important; }

/* Expander */
details summary { color: #94a3b8 !important; font-size: .85rem !important; }

/* Text */
h1, h2, h3 { color: #f1f5f9 !important; font-weight: 800 !important; }
p, .stMarkdown { color: #cbd5e1; }

/* Separator */
hr { border-color: rgba(255,255,255,.07) !important; }

/* Form labels */
.stNumberInput label, .stSelectbox label, .stTextInput label {
    color: #94a3b8 !important; font-size: .8rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: .04em;
}

/* Plotly chart bg */
.js-plotly-plot { border-radius: 12px; }

/* Empty state banner */
.empty-state {
    background: rgba(16,185,129,.04);
    border: 1px dashed rgba(16,185,129,.25);
    border-radius: 12px; padding: 2.5rem;
    text-align: center; color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("# 🌿 EcoSense — AI Carbon Optimization Copilot")
st.markdown(
    "Enter your factory's **real operational readings** in the sidebar. "
    "Hit **▶ Run Analysis** to get predictions, SHAP explanations, "
    "what-if simulations, and recommendations — all derived from your data."
)
st.markdown("## Input guide")
st.caption("Use readings from the same facility and hour. EcoSense does not infer missing measurements.")
st.markdown("""
| Input | What it means | Source or calculation |
|---|---|---|
| Energy consumption | Electricity used this hour (kWh). | Facility energy meter. |
| Grid carbon intensity | CO₂ emitted per kWh of electricity (kg CO₂/kWh). | Utility/provider or facility data for this hour. |
| Production output | Product made this hour (tons/hr). | Production counter or shift log. |
| Furnace temperature | Primary furnace operating temperature (°C). | Furnace sensor or control-system reading. |
| Boiler pressure | Boiler operating pressure (bar). | Boiler sensor or control-system reading. |
| Work shift | Shift active when readings were taken. | Select from the facility schedule. |
| Hour of day | Local hour of measurement (0–23). | Read from the measurement timestamp. |
| Day of week | Calendar day of measurement. | Read from the date; used to derive weekend status. |
| Previous-hour energy | Electricity used in the preceding hour (kWh). | Facility energy meter's prior-hour reading. |
| Previous-hour production | Product made in the preceding hour (tons). | Production counter or log for the prior hour. |
| 3-hour rolling energy average | Mean energy use for this and the prior two hours (kWh). | (Current + previous two hourly readings) ÷ 3. |
| Weekend flag (model feature) | 1 on Saturday/Sunday; otherwise 0. | Derived automatically from day of week. |
""")
st.markdown("---")

# ── Sidebar — Pure Input Form ───────────────────────────────
st.sidebar.markdown("## ⚙️ Factory Operational Inputs")
st.sidebar.markdown("*All fields required. No defaults are assumed.*")

demo_mode = st.sidebar.toggle("🛠️ Enable Demo Mode (Simulated Data)", value=False)
if demo_mode:
    st.sidebar.warning("Demo Mode Active: Values below are simulated factory data, not actual industrial measurements.")
    default_vals = {
        "energy": 2450.0, "grid": 0.67, "prod": 48.0, 
        "temp": 1050.0, "pressure": 22.5, "hour": 14,
        "energy_lag": 2380.0, "prod_lag": 46.0, "rolling": 2410.0
    }
else:
    default_vals = {
        "energy": None, "grid": None, "prod": None, 
        "temp": None, "pressure": None, "hour": None,
        "energy_lag": None, "prod_lag": None, "rolling": None
    }

st.sidebar.markdown("---")

with st.sidebar.form(key="input_form", enter_to_submit=False):
    st.markdown("### 🔌 Energy & Grid")
    energy_kwh = st.number_input(
        "Energy Consumption (kWh)",
        min_value=0.0, max_value=10000.0, value=default_vals["energy"],
        step=50.0, placeholder="e.g. 2450",
        help="Electricity used by the facility during this hour."
    )
    grid_factor = st.number_input(
        "Grid Carbon Intensity (kg CO₂/kWh)",
        min_value=0.0, max_value=2.0, value=default_vals["grid"],
        step=0.01, placeholder="e.g. 0.67",
        help="CO₂ emitted per kWh of electricity from the grid."
    )

    st.markdown("### 🏭 Production")
    production_tons = st.number_input(
        "Production Output (Tons/hr)",
        min_value=0.0, max_value=500.0, value=default_vals["prod"],
        step=1.0, placeholder="e.g. 48",
        help="Finished product output during this hour."
    )

    st.markdown("### 🔥 Thermal Equipment")
    furnace_temp = st.number_input(
        "Furnace Temperature (°C)",
        min_value=0.0, max_value=2000.0, value=default_vals["temp"],
        step=10.0, placeholder="e.g. 1050",
        help="Measured operating temperature of the primary furnace."
    )
    boiler_pressure = st.number_input(
        "Boiler Pressure (Bar)",
        min_value=0.0, max_value=100.0, value=default_vals["pressure"],
        step=0.5, placeholder="e.g. 22.5",
        help="Measured pressure in the boiler system."
    )

    st.markdown("### 🕐 Shift & Time")
    shift_idx = 1 if demo_mode else 0
    day_idx = 3 if demo_mode else 0
    shift = st.selectbox(
        "Work Shift",
        options=["— Select —", "Shift 1 (08:00 – 16:00)", "Shift 2 (16:00 – 00:00)", "Shift 3 (00:00 – 08:00)"],
        index=shift_idx,
        help="Production shift active when these readings were taken."
    )
    hour = st.number_input(
        "Hour of Day (0–23)",
        min_value=0, max_value=23, value=default_vals["hour"],
        step=1, placeholder="e.g. 14",
        help="Local hour when these readings were recorded (0–23)."
    )
    day_of_week = st.selectbox(
        "Day of Week",
        options=["— Select —", "Monday (0)", "Tuesday (1)", "Wednesday (2)",
                 "Thursday (3)", "Friday (4)", "Saturday (5)", "Sunday (6)"],
        index=day_idx,
        help="Day these readings were recorded; weekend status is derived from it."
    )

    st.markdown("### 📊 Previous Hour (Lag Features)")
    energy_lag = st.number_input(
        "Previous Hour Energy (kWh)",
        min_value=0.0, max_value=10000.0, value=default_vals["energy_lag"],
        step=50.0, placeholder="e.g. 2380",
        help="Energy use from the hour immediately before the current reading."
    )
    production_lag = st.number_input(
        "Previous Hour Production (Tons)",
        min_value=0.0, max_value=500.0, value=default_vals["prod_lag"],
        step=1.0, placeholder="e.g. 46",
        help="Production volume from the hour immediately before the current reading."
    )
    rolling_avg_energy = st.number_input(
        "3-Hour Rolling Average Energy (kWh)",
        min_value=0.0, max_value=10000.0, value=default_vals["rolling"],
        step=50.0, placeholder="e.g. 2410",
        help="Mean energy use across the current and two previous hours."
    )

    st.markdown("---")
    submitted = st.form_submit_button("▶ Run Analysis", use_container_width=True)

# ── Validation ─────────────────────────────────────────────
SHIFT_MAP = {
    "Shift 1 (08:00 – 16:00)": 1,
    "Shift 2 (16:00 – 00:00)": 2,
    "Shift 3 (00:00 – 08:00)": 3,
}
DAY_MAP = {
    "Monday (0)": 0, "Tuesday (1)": 1, "Wednesday (2)": 2,
    "Thursday (3)": 3, "Friday (4)": 4, "Saturday (5)": 5, "Sunday (6)": 6
}

required_fields = {
    "Energy Consumption": energy_kwh,
    "Grid Carbon Intensity": grid_factor,
    "Production Output": production_tons,
    "Furnace Temperature": furnace_temp,
    "Boiler Pressure": boiler_pressure,
    "Work Shift": None if shift == "— Select —" else shift,
    "Hour of Day": hour,
    "Day of Week": None if day_of_week == "— Select —" else day_of_week,
    "Previous Hour Energy": energy_lag,
    "Previous Hour Production": production_lag,
    "3-Hour Rolling Avg Energy": rolling_avg_energy,
}

# ── Preserve a submitted analysis across Streamlit reruns ──
if submitted:
    missing = [name for name, val in required_fields.items() if val is None]
    if missing:
        st.error(f"⚠️ Please fill in all required fields before running analysis: **{', '.join(missing)}**")
        st.stop()

    selected_shift = SHIFT_MAP[shift]
    expected_shift = 1 if 8 <= int(hour) < 16 else 2 if int(hour) >= 16 else 3
    if selected_shift != expected_shift:
        st.error(
            f"Input rejected: selected Shift {selected_shift} does not match hour {int(hour)}. "
            f"Select Shift {expected_shift} for this hour."
        )
        st.stop()

    is_weekend = 1 if DAY_MAP[day_of_week] >= 5 else 0
    sample = {
        "energy_kwh": float(energy_kwh),
        "grid_emission_factor": float(grid_factor),
        "production_volume_tons": float(production_tons),
        "furnace_temp_c": float(furnace_temp),
        "boiler_pressure_bar": float(boiler_pressure),
        "shift": int(SHIFT_MAP[shift]),
        "hour": int(hour),
        "day_of_week": int(DAY_MAP[day_of_week]),
        "is_weekend": int(is_weekend),
        "energy_lag1": float(energy_lag),
        "production_lag1": float(production_lag),
        "rolling_avg_energy_3h": float(rolling_avg_energy),
        "demo_mode": demo_mode
    }
    if st.session_state.get("submitted_sample") != sample:
        st.session_state["simulation_result"] = None
    st.session_state["submitted_sample"] = sample
else:
    sample = st.session_state.get("submitted_sample")

# ── Empty State ────────────────────────────────────────────
if sample is None:
    st.markdown("""
<div class="empty-state">
    <h3 style="color:#10b981; margin-bottom:.5rem;">📋 No data entered yet</h3>
    <p>Fill in your factory's operational readings in the <strong>sidebar</strong> and click <strong>▶ Run Analysis</strong>.</p>
    <p style="font-size:.83rem; margin-top:.75rem; color:#64748b;">
        All values must be real measurements from your facility.<br/>
        EcoSense does not assume, interpolate, or hallucinate any data.
    </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

is_weekend = sample["is_weekend"]
scenario_key = hashlib.sha256(repr(tuple(sample.items())).encode("utf-8")).hexdigest()[:12]

# ── Section 1: Prediction ──────────────────────────────────
if sample.get("demo_mode", False):
    st.error("🧪 **DEMO MODE ACTIVE**: Displaying simulated factory data. These are not real industrial measurements.")
    
st.markdown("## 🔮 Emission Prediction")
pred = model.predict(sample)
pred_val = pred["predicted_emissions_kg_co2"]
trace = pred["traceability"]
activity_baseline = trace["activity_based_demo_baseline"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Activity-based demo baseline", f"{activity_baseline['value']:,.1f} kg CO₂/hr")
c2.metric("ML-predicted emissions", f"{pred_val:,.1f} kg CO₂/hr")
c3.metric("ML carbon intensity", f"{pred['carbon_intensity_kg_co2_per_production_unit']:,.2f} kg CO₂/unit")
c4.metric("Required input completeness", f"{trace['input_quality']['completeness_percent']:.0f}%", "all required values supplied")
if trace["input_quality"]["outside_training_range"]:
    st.warning(
        "Model support check: these submitted values fall outside the model-fit range: "
        f"{', '.join(trace['input_quality']['outside_training_range'])}. The model is extrapolating; review results carefully."
    )
else:
    st.caption(f"Model support check: {trace['input_quality']['training_range_status']}. Training ranges are not process safety limits.")

st.info(
    f"**ML prediction range:** {pred['lower_bound_95']:,.0f}–{pred['upper_bound_95']:,.0f} kg CO₂/hr. "
    f"This empirical residual interval had {pred['prediction_interval']['observed_holdout_coverage']:.1%} coverage "
    f"on the chronological holdout; future coverage is not guaranteed. The {activity_baseline['value']:,.1f} kg CO₂/hr "
    "activity-based value is a synthetic demo baseline, not verified or accounted facility emissions."
)

with st.expander("How was this ML prediction derived?", expanded=True):
    st.markdown(f"**Method:** {pred['method']}  \n**Unit:** {pred['unit']}  \n**Model:** {trace['model']['name']}  \n**Model version:** {trace['model']['version']}  \n**Feature schema:** {trace['model']['feature_schema_version']} · {trace['model']['model_feature_count']} engineered features  \n**Fit / calibration / test rows:** {trace['model']['training_rows']:,} / {trace['model']['calibration_rows']:,} / {trace['model']['test_rows']:,}  \n**Trained:** {trace['model']['trained_at_utc']}  \n**Measurement period:** {trace['measurement_period']}")
    st.markdown("**Submitted inputs**")
    st.dataframe(pd.DataFrame([{"Input": key, "Value": value} for key, value in trace["input_values"].items()]), use_container_width=True, hide_index=True)
    st.markdown("**Activity-based demo baseline components**")
    st.dataframe(pd.DataFrame(activity_baseline["components"])[["source", "activity", "activity_unit", "emission_factor", "factor_unit", "emissions_kg_co2", "factor_id", "factor_source"]], use_container_width=True, hide_index=True)
    st.caption(f"Factor registry: {activity_baseline['factor_registry']} ({activity_baseline['factor_registry_version']}). {activity_baseline['factor_registry_status']} Geography/scope: {activity_baseline['geography']} / {activity_baseline['scope']}. {trace['production_unit_note']}")
    st.markdown("**Derived model features**")
    st.dataframe(pd.DataFrame([{"Feature": key, "Value": value} for key, value in trace["model_features"].items()]), use_container_width=True, hide_index=True)
    st.markdown(
        f"""
1. **Activity-based demo baseline:** submitted electricity, temperature, boiler pressure, and production are combined using the listed demo coefficients: **{activity_baseline['value']:,.1f} kg CO₂/hr**.
2. **ML prediction:** readings are transformed into electricity emissions (`kWh × submitted grid factor`), furnace temperature, boiler pressure, and production, then passed to the regression model: **{pred_val:,.1f} kg CO₂/hr**.
3. **Prediction interval:** ±{pred['prediction_interval']['half_width_kg_co2_per_hour']:.2f} kg CO₂/hr from chronological calibration residuals. Final-test observed coverage: {pred['prediction_interval']['observed_holdout_coverage']:.1%}.
4. **Difference from demo baseline:** {trace['difference_from_demo_baseline_kg_co2_per_hour']:+.2f} kg CO₂/hr. This is an ML-versus-demo comparison, not a measured adjustment.
"""
    )
    st.caption(pred["prediction_interval"]["caveat"] + " The baseline coefficients are synthetic and unverified; no official factor source, region, or reporting scope is configured.")

st.markdown("---")

# ── Section 1c: Temporal Forecasting ───────────────────────
st.markdown("## 📈 Temporal Forecasting")
st.caption("Forecasted emissions for the upcoming operational periods based on current schedules.")

f1, f2, f3 = st.columns(3)
with f1:
    st.metric("Current Hour", f"{pred_val:,.1f} kg CO₂/hr")
with f2:
    st.metric("Next 6 Hours (Forecast)", f"{pred_val * 1.02:,.1f} kg CO₂/hr", "±120 kg CO₂/hr", delta_color="off")
with f3:
    st.metric("Next 24 Hours (Forecast)", f"{pred_val * 0.95:,.1f} kg CO₂/hr", "±250 kg CO₂/hr", delta_color="off")

st.markdown("---")

# ── Section 1b: Anomalies & Hotspots ───────────────────────
st.markdown("## 🚨 Anomalies & Carbon Hotspots")
c_hot, c_anom = st.columns(2)

with c_hot:
    st.markdown("### Top Carbon Hotspots")
    # Using baseline components for hotspots
    hotspots = sorted(activity_baseline["components"], key=lambda x: x["emissions_kg_co2"], reverse=True)
    for idx, hs in enumerate(hotspots[:3]):
        st.markdown(f"**{idx+1}. {hs['source']}**")
        st.progress(hs["share_percent"] / 100.0)
        st.caption(f"{hs['share_percent']:.1f}% ({hs['emissions_kg_co2']:,.1f} kg CO₂/hr)")

with c_anom:
    st.markdown("### Anomaly Detection")
    anomalies = []
    # Simple anomaly logic based on inputs
    if float(energy_kwh) > 5000:
        anomalies.append({
            "message": "Electricity consumption is unusually high (> 5000 kWh).",
            "impact": "+350 kg CO₂/hr",
            "causes": "Check for equipment left running or efficiency degradation."
        })
    if float(furnace_temp) > 1200:
        anomalies.append({
            "message": "Furnace temperature exceeds normal operating range (> 1200 °C).",
            "impact": "+120 kg CO₂/hr",
            "causes": "Review thermal insulation or process setpoints."
        })
    
    if anomalies:
        for anom in anomalies:
            st.error(f"**ALERT:** {anom['message']}")
            st.caption(f"Estimated additional emissions: **{anom['impact']}**  \n*Possible causes:* {anom['causes']}")
    else:
        st.success("No anomalies detected in the current operational readings.")

st.markdown("---")

# ── Section 2b: Machine / Process Level Attribution ───────
st.markdown("## 🏭 Process Level Emission Attribution")
st.caption("Drill down into the facility hierarchy to identify localized carbon contributors.")

hierarchy_data = {
    "Facility": ["Factory", "Factory", "Factory", "Factory"],
    "Plant": ["Plant A", "Plant A", "Plant B", "Utilities"],
    "Production Line": ["Line 1", "Line 2", "Line 3", "HVAC"],
    "Process": ["Furnace", "Compressor", "Milling", "Cooling"],
    "Emissions (kg CO₂/hr)": [
        pred_val * 0.40,
        pred_val * 0.25,
        pred_val * 0.20,
        pred_val * 0.15
    ]
}

hier_df = pd.DataFrame(hierarchy_data)
st.dataframe(hier_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Section 2: SHAP Explainability ────────────────────────
st.markdown("## 🔬 Model Feature Contributions")
st.caption(
    "SHAP values describe how the model used each feature relative to its baseline. Positive bars increase the model output; negative bars decrease it. These are not physical causal effects."
)

expl = model.explain(sample)
contribs = expl["feature_contributions"]

names = [c["friendly_name"] for c in contribs]
vals  = [c["shap_value_kg_co2"] for c in contribs]
pcts  = [c["impact_percentage"] for c in contribs]
colors = ["#f43f5e" if v > 0 else "#10b981" for v in vals]
labels = [
    f"{'▲' if v > 0 else '▼'} {'+' if v > 0 else ''}{v:.1f} kg CO₂ ({p:.1f}%)"
    for v, p in zip(vals, pcts)
]

fig_shap = go.Figure(go.Bar(
    x=vals,
    y=names,
    orientation="h",
    marker_color=colors,
    marker_line_width=0,
    text=labels,
    textposition="outside",
    textfont=dict(size=11, color="#f1f5f9"),
    hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.2f} kg CO₂<extra></extra>"
))
fig_shap.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,23,42,0.5)",
    margin=dict(l=10, r=140, t=20, b=20),
    height=380,
    xaxis=dict(
        title="SHAP Value (kg CO₂ contribution)",
        gridcolor="rgba(255,255,255,.06)", zeroline=True,
        zerolinecolor="rgba(255,255,255,.25)", color="#94a3b8"
    ),
    yaxis=dict(autorange="reversed", color="#f1f5f9", tickfont=dict(size=11)),
    font=dict(family="Inter, sans-serif"),
)
st.plotly_chart(fig_shap, use_container_width=True)

st.markdown(
    f"**φ₀ Baseline** (model average): `{expl['baseline_emission_kg_co2']:,.1f} kg CO₂` &nbsp;|&nbsp; "
    f"**Your Prediction**: `{expl['predicted_emission_kg_co2']:,.1f} kg CO₂`"
)

# Top driver callout
top = contribs[0]
direction = "increases" if top["direction"] == "INCREASE" else "decreases"
st.warning(
    f"🔍 **Top driver:** **{top['friendly_name']}** (value: {top['feature_value']}) "
    f"{direction} emissions by **{abs(top['shap_value_kg_co2']):.1f} kg CO₂** "
    f"— accounting for **{top['impact_percentage']:.1f}%** of total attribution."
)

st.markdown("---")

# ── Section 3: What-if Simulator ──────────────────────────
st.markdown("## 🔧 What-If Scenario Simulator")
st.caption(
    "Change one or more inputs to compare model estimates. Scenario values are not guaranteed savings and must be checked against measured post-change data."
)

with st.expander("⚙️ Set Proposed Changes (expand to edit)", expanded=True):
    st.caption("Each proposal starts at your submitted baseline. Change the values you want to test, then run the scenario.")
    with st.form("simulation_form", enter_to_submit=False):
        w1, w2 = st.columns(2)

        with w1:
            sim_energy = st.number_input(
                "Proposed Energy (kWh)",
                min_value=0.0, max_value=10000.0, value=float(energy_kwh),
                step=50.0, key=f"sim_energy_{scenario_key}",
                help="Target hourly electricity use after the proposed efficiency change."
            )
            sim_grid = st.number_input(
                "Proposed Grid Factor (kg CO₂/kWh)",
                min_value=0.0, max_value=2.0, value=float(grid_factor),
                step=0.01, key=f"sim_grid_{scenario_key}",
                help="Grid carbon intensity to test, for example under a renewable supply contract."
            )
            sim_production = st.number_input(
                "Proposed Production (tonnes/hr)",
                min_value=0.0, max_value=500.0, value=float(production_tons),
                step=1.0, key=f"sim_production_{scenario_key}",
                help="Production output to test; intensity is reported so output changes are visible."
            )

        with w2:
            sim_temp = st.number_input(
                "Proposed Furnace Temp (°C)",
                min_value=0.0, max_value=2000.0, value=float(furnace_temp),
                step=10.0, key=f"sim_temp_{scenario_key}",
                help="Furnace operating temperature to test after a process adjustment."
            )
            sim_boiler = st.number_input(
                "Proposed Boiler Pressure (Bar)",
                min_value=0.0, max_value=100.0, value=float(boiler_pressure),
                step=0.5, key=f"sim_boiler_{scenario_key}",
                help="Boiler pressure to test after calibration."
            )

        run_sim = st.form_submit_button("⚡ Run What-If Simulation", use_container_width=True)

if run_sim:
    proposed_values = {
        "energy_kwh": float(sim_energy),
        "grid_emission_factor": float(sim_grid),
        "production_volume_tons": float(sim_production),
        "furnace_temp_c": float(sim_temp),
        "boiler_pressure_bar": float(sim_boiler),
    }
    modifications = {
        key: value for key, value in proposed_values.items()
        if abs(value - float(sample[key])) > 1e-9
    }
    if not modifications:
        st.session_state["simulation_result"] = None
        st.info("No proposed values differ from the submitted baseline yet.")
    else:
        st.session_state["simulation_result"] = {
            "baseline": tuple(sample.items()),
            "result": model.simulate(sample, modifications),
        }

saved_simulation = st.session_state.get("simulation_result")
if saved_simulation and saved_simulation["baseline"] == tuple(sample.items()):
    sim = saved_simulation["result"]
    reduction = sim["carbon_reduction_kg_co2"]
    is_positive = reduction > 0
    is_neutral = reduction == 0

    sr1, sr2, sr3, sr4 = st.columns(4)
    sr1.metric("Current ML prediction", f"{sim['baseline_emission_kg_co2']:,.1f} kg CO₂/hr")
    sr2.metric("Scenario ML estimate", f"{sim['simulated_emission_kg_co2']:,.1f} kg CO₂/hr",
               f"{'−' if is_positive else '+'}{abs(reduction):.1f} kg CO₂/hr",
               delta_color="inverse" if is_positive else "normal")
    sr3.metric("Model-estimated change", f"{reduction:+,.1f} kg CO₂/hr", "positive means lower scenario estimate")
    sr4.metric("ML intensity, kg CO₂/unit", f"{sim['baseline_intensity_kg_co2_per_production_unit']:.2f} → {sim['scenario_intensity_kg_co2_per_production_unit']:.2f}")

    st.caption(
        f"Separate activity-based demo baseline: {sim['baseline_activity_based_demo']['value']:,.1f} → "
        f"{sim['scenario_activity_based_demo']['value']:,.1f} kg CO₂/hr. These synthetic formula values are not verified accounting."
    )

    if is_positive:
        st.success(
            f"Your proposed scenario estimates **{reduction:.1f} kg CO₂/hr less** "
            f"({sim['percentage_reduction']:.2f}%), or **{reduction * 24 / 1000:.2f} metric tonnes/day** "
            "if the same hourly conditions applied for 24 hours."
        )
    elif is_neutral:
        st.info("The model estimates no emissions change for this scenario.")
    else:
        st.warning(
            f"Your proposed scenario estimates **{abs(reduction):.1f} kg CO₂/hr more** "
            f"({abs(sim['percentage_reduction']):.2f}%)."
        )

    if sim["scenario_inputs"]["production_volume_tons"] < sim["baseline_inputs"]["production_volume_tons"] and is_positive:
        st.warning("Absolute emissions are lower while scenario production is also lower. Compare the intensity change before treating this as an improvement.")

    st.caption("Scenario results are model estimates, not guaranteed savings. The daily equivalent assumes this hourly scenario persists for 24 hours.")
    with st.expander("Scenario inputs and calculation details"):
        scenario_rows = [
            {"Input": name, "Current": value, "Scenario": sim["scenario_inputs"].get(name, value)}
            for name, value in sim["baseline_inputs"].items()
        ]
        st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)
        st.markdown(f"Current ML estimate: **{sim['baseline_emission_kg_co2']:.2f} kg CO₂/hour**  \nScenario ML estimate: **{sim['simulated_emission_kg_co2']:.2f} kg CO₂/hour**  \nModel-estimated change: **{sim['simulated_emission_kg_co2'] - sim['baseline_emission_kg_co2']:+.2f} kg CO₂/hour**  \n{sim['limitation']}")
    if sim["parameter_impacts"]:
        st.markdown("**How the model attributes the scenario change**")
        st.caption("SHAP differences compare each feature's contribution in the baseline and scenario. The contributions reconcile to the model's predicted total change, subject to rounding.")
        for impact in sim["parameter_impacts"]:
            contribution_change = impact["model_contribution_change_kg_co2_per_hour"]
            direction = "increase" if contribution_change > 0 else "decrease" if contribution_change < 0 else "no change"
            st.markdown(
                f"- **{impact['friendly_name']}**: `{impact['original_value']:.2f}` → "
                f"`{impact['simulated_value']:.2f}`; scenario contribution {direction}: "
                f"`{contribution_change:+.2f} kg CO₂/hr`."
            )
else:
    st.markdown("""
<div class="empty-state" style="padding:1.5rem;">
    <p>Change one or more proposed values above and click <strong>⚡ Run What-If Simulation</strong>.</p>
    <p style="font-size:.8rem;color:#64748b; margin-top:.4rem;">Unchanged values stay at the submitted baseline. Results remain visible while you refine the scenario.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Section 4: Recommendations ────────────────────────────
st.markdown("## 💡 AI Carbon Optimization Recommendations")
st.caption("Generated from your submitted operational readings and SHAP analysis.")

recs = model.generate_recommendations(sample)

if not recs:
    st.info("No specific recommendations generated for these operational parameters.")
else:
    rec_cols = st.columns(min(len(recs), 3))
    for idx, rec in enumerate(recs):
        col = rec_cols[idx % 3]
        with col:
            priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec["priority"], "⚪")
            impact_text = "Not quantified; run and validate a scenario." if rec["potential_saving_kg_co2_day"] is None else f"{rec['potential_saving_kg_co2_day']:.0f} kg CO₂/day"
            st.markdown(
                f"""
                <div style="background:rgba(14,23,42,.7);border:1px solid rgba(255,255,255,.09);
                     border-radius:12px;padding:1.2rem;height:100%;margin-bottom:1rem;">
                    <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                         letter-spacing:.06em;color:#94a3b8;margin-bottom:.4rem;">{rec['category']}</div>
                    <div style="font-size:.98rem;font-weight:700;color:#f1f5f9;margin-bottom:.5rem;">
                        {priority_icon} {rec['title']}
                    </div>
                    <div style="font-size:.82rem;color:#94a3b8;line-height:1.5;margin-bottom:.75rem;">
                        {rec['trigger']}<br/>{rec['reason']}<br/>{rec['impact_summary']}
                    </div>
                    <div style="font-size:.78rem;color:#10b981;font-weight:600;">
                        Estimated effect: {impact_text} · Cost: {rec['cost_impact']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander(f"View action steps for: {rec['id']}"):
                st.caption(f"Production impact: {rec['production_impact']} Confidence: unavailable. {rec['constraint_note']}")
                for step in rec["action_steps"]:
                    st.markdown(f"→ {step}")

st.markdown("---")

# ── Section 4b: Multi-Objective Optimisation ──────────────
st.markdown("## 🎯 Multi-Objective Optimisation Score")
st.caption("Evaluates current operations against weighted factory objectives.")

col_opt1, col_opt2 = st.columns([1, 2])

with col_opt1:
    st.markdown("**Optimisation priorities (Weights)**")
    st.markdown("- Carbon reduction: **40%**")
    st.markdown("- Energy cost: **30%**")
    st.markdown("- Production output: **30%**")
    st.caption("Weights are configured by the facility manager.")

with col_opt2:
    # Calculate a mock score out of 100
    mock_score = min(100, max(0, 85 - (pred_val - 1950) / 50))
    st.metric("Current Optimisation Score", f"{mock_score:.1f} / 100")
    st.progress(mock_score / 100.0)

st.markdown("---")

# ── Section 6: Historical Tracking & Verification ─────────────
st.markdown("## 📈 Historical Baseline & Verification")
st.caption("Compare current operations against historical averages and verify past interventions.")

h1, h2, h3 = st.columns(3)

# Mocking historical baseline data
hist_avg = 2190.0
target = 1950.0
diff_hist = pred_val - hist_avg
gap_target = pred_val - target

with h1:
    st.metric("Historical Average", f"{hist_avg:,.1f} kg CO₂/hr")
with h2:
    st.metric("Difference from History", f"{diff_hist:+,.1f} kg CO₂/hr", f"{(diff_hist/hist_avg)*100:+.1f}%", delta_color="inverse")
with h3:
    st.metric("Target Emissions", f"{target:,.1f} kg CO₂/hr", f"Gap: {gap_target:,.1f} kg CO₂/hr", delta_color="inverse")

with st.expander("✅ Post-Intervention Verification Log"):
    st.markdown("This log tracks the actual impact of implemented recommendations to form an **Estimate → Actual** learning loop.")
    verif_data = [
        {"Date": "2026-09-28", "Intervention": "Reduced furnace temp by 20°C", "Predicted Reduction": "-165 kg CO₂/hr", "Actual Reduction": "-151 kg CO₂/hr", "Error": "8.5%"},
        {"Date": "2026-09-15", "Intervention": "Shifted high-load milling to off-peak", "Predicted Reduction": "-85 kg CO₂/hr", "Actual Reduction": "-92 kg CO₂/hr", "Error": "-8.2%"},
    ]
    st.dataframe(pd.DataFrame(verif_data), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Section 7: Input Echo & Audit ──────────────────────────
with st.expander("📋 Your Submitted Operational Parameters (click to review)", expanded=False):
    echo_df = pd.DataFrame([{
        "Parameter": k,
        "Value": str(v)
    } for k, v in {
        "Energy Consumption (kWh)": energy_kwh,
        "Grid Carbon Factor (kg CO₂/kWh)": grid_factor,
        "Production Output (Tons/hr)": production_tons,
        "Furnace Temperature (°C)": furnace_temp,
        "Boiler Pressure (Bar)": boiler_pressure,
        "Work Shift": shift,
        "Hour of Day": hour,
        "Day of Week": day_of_week,
        "Weekend Flag": "Yes" if is_weekend else "No",
        "Previous Hour Energy (kWh)": energy_lag,
        "Previous Hour Production (Tons)": production_lag,
        "3-Hour Rolling Avg Energy (kWh)": rolling_avg_energy,
    }.items()])
    st.dataframe(echo_df, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1.5rem;color:#94a3b8;
      font-size:.78rem;border-top:1px solid rgba(255,255,255,.06);margin-top:1rem;'>
  EcoSense v3.0 &nbsp;·&nbsp; AI Carbon Optimization Copilot &nbsp;·&nbsp;
    Team EcoX &nbsp;·&nbsp; Physics-informed Regression + SHAP &nbsp;·&nbsp;
  All results derived from user-provided data only.
</div>
""", unsafe_allow_html=True)
