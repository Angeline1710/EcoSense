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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --app-ground: #f1f3ed;
    --app-surface: #ffffff;
    --app-ink: #26332b;
    --app-muted: #58645c;
    --app-rule: #ccd4ca;
    --app-olive: #3f6249;
    --app-oxide: #a44c37;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; }
.stApp { background: var(--app-ground) !important; color: var(--app-ink) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #e7ebe4 !important;
    border-right: 1px solid var(--app-rule) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: var(--app-muted) !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--app-surface);
    border: 1px solid var(--app-rule);
    border-radius: 6px;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label { color: var(--app-muted) !important; font-size: 0.76rem !important; text-transform: uppercase; letter-spacing: .05em; }
div[data-testid="stMetricValue"] { color: var(--app-ink) !important; font-weight: 600 !important; font-size: 1.5rem !important; }

/* Input fields */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #bfc9bd !important;
    color: var(--app-ink) !important;
    border-radius: 4px !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--app-olive) !important;
    box-shadow: 0 0 0 1px var(--app-olive) !important;
}
input { color: var(--app-ink) !important; }

/* Buttons */
.stButton > button {
    background: var(--app-olive) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: .9rem !important;
    padding: .6rem 1.5rem !important;
    box-shadow: none !important;
    transition: background-color .15s ease !important;
}
.stButton > button:hover { background: #314c39 !important; }

/* Alert/info boxes */
.stAlert { border-radius: 10px !important; }

/* Expander */
details summary { color: var(--app-olive) !important; font-size: .85rem !important; }

/* Text */
h1, h2, h3 { color: var(--app-ink) !important; font-weight: 600 !important; }
p, .stMarkdown { color: var(--app-muted); }

/* Separator */
hr { border-color: var(--app-rule) !important; }

/* Form labels */
.stNumberInput label, .stSelectbox label, .stTextInput label {
    color: var(--app-muted) !important; font-size: .8rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: .04em;
}

/* Plotly chart bg */
.js-plotly-plot { border-radius: 12px; }

/* Empty state banner */
.empty-state {
    background: #f8faf6;
    border: 1px dashed var(--app-rule);
    border-radius: 12px; padding: 2.5rem;
    text-align: center; color: var(--app-muted);
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("# 🌿 EcoSense — AI Carbon Optimization Copilot")
st.markdown(
    "Enter five measured operating inputs in the sidebar. "
    "The active model estimates current emissions, shows model contributions, "
    "and compares explicitly submitted what-if scenarios."
)
st.markdown("## Input guide")
st.caption("Required model inputs must describe the same operating interval. Advanced context is optional and is not used by the current estimator.")
st.markdown("""
| Input | What it means | Source or calculation |
|---|---|---|
| Energy consumption | Electricity used this hour (kWh). | Facility energy meter. |
| Grid carbon intensity | CO₂ emitted per kWh of electricity (kg CO₂/kWh). | Utility/provider or facility data for this hour. |
| Production output | Product made this hour (tons/hr). | Production counter or shift log. |
| Furnace temperature | Primary furnace operating temperature (°C). | Furnace sensor or control-system reading. |
| Boiler pressure | Boiler operating pressure (bar). | Boiler sensor or control-system reading. |
""")
st.markdown("---")

# ── Sidebar — Pure Input Form ───────────────────────────────
st.sidebar.markdown("## ⚙️ Factory Operational Inputs")
st.sidebar.markdown("*Five measured fields required for the active model. No measured values are assumed.*")

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

    with st.expander("Advanced / experimental context (not used by the active model)", expanded=demo_mode):
        st.caption("Optional shift, calendar, lag, and rolling values are retained for traceability only; they do not affect this model's estimate.")
        shift_idx = 1 if demo_mode else 0
        day_idx = 3 if demo_mode else 0
        shift = st.selectbox(
            "Work Shift",
            options=["— Select —", "Shift 1 (08:00 – 16:00)", "Shift 2 (16:00 – 00:00)", "Shift 3 (00:00 – 08:00)"],
            index=shift_idx,
            help="Optional shift context; if supplied, provide the matching hour."
        )
        hour = st.number_input(
            "Hour of Day (0–23)",
            min_value=0, max_value=23, value=default_vals["hour"],
            step=1, placeholder="optional",
            help="Optional local hour context; not used by the active model."
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=["— Select —", "Monday (0)", "Tuesday (1)", "Wednesday (2)",
                     "Thursday (3)", "Friday (4)", "Saturday (5)", "Sunday (6)"],
            index=day_idx,
            help="Optional calendar context; not used by the active model."
        )
        energy_lag = st.number_input(
            "Previous Hour Energy (kWh)", min_value=0.0, max_value=10000.0,
            value=default_vals["energy_lag"], step=50.0, placeholder="optional",
            help="Optional experimental lag; not used by the active model."
        )
        production_lag = st.number_input(
            "Previous Hour Production (Tons)", min_value=0.0, max_value=500.0,
            value=default_vals["prod_lag"], step=1.0, placeholder="optional",
            help="Optional experimental lag; not used by the active model."
        )
        rolling_avg_energy = st.number_input(
            "3-Hour Rolling Average Energy (kWh)", min_value=0.0, max_value=10000.0,
            value=default_vals["rolling"], step=50.0, placeholder="optional",
            help="Optional experimental rolling feature; not used by the active model."
        )

    st.markdown("---")
    submitted = st.form_submit_button("▶ Run Analysis", width="stretch")

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
}

# ── Preserve a submitted analysis across Streamlit reruns ──
if submitted:
    missing = [name for name, val in required_fields.items() if val is None]
    if missing:
        st.error(f"⚠️ Please fill in all required fields before running analysis: **{', '.join(missing)}**")
        st.stop()

    shift_supplied = shift != "— Select —"
    hour_supplied = hour is not None
    if shift_supplied != hour_supplied:
        st.error("Input rejected: provide both shift and hour, or leave both unset.")
        st.stop()
    if shift_supplied:
        selected_shift = SHIFT_MAP[shift]
        expected_shift = 1 if 8 <= int(hour) < 16 else 2 if int(hour) >= 16 else 3
        if selected_shift != expected_shift:
            st.error(
                f"Input rejected: selected Shift {selected_shift} does not match hour {int(hour)}. "
                f"Select Shift {expected_shift} for this hour."
            )
            st.stop()

    sample = {
        "energy_kwh": float(energy_kwh),
        "grid_emission_factor": float(grid_factor),
        "production_volume_tons": float(production_tons),
        "furnace_temp_c": float(furnace_temp),
        "boiler_pressure_bar": float(boiler_pressure),
        "demo_mode": demo_mode
    }
    if shift_supplied:
        sample.update({"shift": int(SHIFT_MAP[shift]), "hour": int(hour)})
    if day_of_week != "— Select —":
        sample["day_of_week"] = int(DAY_MAP[day_of_week])
        sample["is_weekend"] = int(DAY_MAP[day_of_week] >= 5)
    optional_inputs = {
        "energy_lag1": energy_lag,
        "production_lag1": production_lag,
        "rolling_avg_energy_3h": rolling_avg_energy,
    }
    sample.update({key: float(value) for key, value in optional_inputs.items() if value is not None})
    if st.session_state.get("submitted_sample") != sample:
        st.session_state["simulation_result"] = None
    st.session_state["submitted_sample"] = sample
else:
    sample = st.session_state.get("submitted_sample")

# ── Empty State ────────────────────────────────────────────
if sample is None:
    empty_state_note = (
        "Demo mode is active; these are simulated values, not facility observations."
        if demo_mode
        else "Use readings from your facility; source authenticity is not independently verified."
    )
    st.markdown(f"""
<div class="empty-state">
    <h3 style="color:#10b981; margin-bottom:.5rem;">📋 No data entered yet</h3>
    <p>Fill in your factory's operational readings in the <strong>sidebar</strong> and click <strong>▶ Run Analysis</strong>.</p>
    <p style="font-size:.83rem; margin-top:.75rem; color:#64748b;">
        {empty_state_note}<br/>
        EcoSense does not assume, interpolate, or hallucinate any data.
    </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

is_weekend = sample.get("is_weekend")
scenario_key = hashlib.sha256(repr(tuple(sample.items())).encode("utf-8")).hexdigest()[:12]

# ── Section 1: Prediction ──────────────────────────────────
if sample.get("demo_mode", False):
    st.error("🧪 **DEMO MODE ACTIVE**: Displaying simulated factory data. These are not real industrial measurements.")
    
st.markdown("## 🔮 Emission Prediction")
pred = model.predict(sample)
validity = pred["prediction_validity"]
if not validity["prediction_available"]:
    st.error("Prediction unavailable for this operating condition.")
    st.caption(f"{validity['reason']} Review the reported model-support range details before submitting another estimate.")
    st.json(pred["model_support"])
    st.stop()
if validity["status"] == "DEGRADED":
    st.warning(f"Model extrapolation is outside observed training conditions. {validity['reason']}")

pred_val = pred["predicted_emissions_kg_co2"]
trace = pred["traceability"]
activity_baseline = trace["activity_reference"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Synthetic Activity Reference", f"{activity_baseline['value']:,.1f} kg CO₂/hr")
c2.metric("ML-predicted emissions", f"{pred_val:,.1f} kg CO₂/hr")
c3.metric("ML carbon intensity", f"{pred['carbon_intensity_kg_co2_per_production_unit']:,.2f} kg CO₂/submitted unit")
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
    "Synthetic Activity Reference is a demo formula, not verified or accounted facility emissions."
)

with st.expander("How reliable is this estimate?"):
    st.markdown(f"**Prediction validity:** {validity['status']} · {validity['reason']}")
    st.markdown(f"**Model support:** {pred['model_support']['status']} · affected features: {', '.join(pred['model_support']['affected_features']) or 'none'}")
    st.markdown(f"**Required-input completeness:** {pred['data_quality']['completeness_percent']:.0f}%. This checks field presence only; it is not sensor/data authenticity or model confidence.")
    st.markdown(f"**Empirical interval:** {pred['prediction_interval']['method']} · final-test observed coverage {pred['prediction_interval']['observed_holdout_coverage']:.1%} on synthetic data.")
    st.markdown(f"**Validation:** MAE {trace['model']['test_mae_kg_co2_per_hour']:.2f} kg CO₂/hr · RMSE {trace['model']['test_rmse_kg_co2_per_hour']:.2f} kg CO₂/hr · R² {trace['model']['test_r2']:.4f}. These scores are not real-factory validation.")
    st.caption("Timestamp freshness, sensor anomalies, drift, and independent measurement verification are not assessed.")

with st.expander("How was this ML prediction derived?", expanded=True):
    st.markdown(f"**Method:** {pred['method']}  \n**Unit:** {pred['unit']}  \n**Model:** {trace['model']['name']}  \n**Model version:** {trace['model']['version']}  \n**Feature schema:** {trace['model']['feature_schema_version']} · {trace['model']['model_feature_count']} engineered features  \n**Fit / calibration / test rows:** {trace['model']['training_rows']:,} / {trace['model']['calibration_rows']:,} / {trace['model']['test_rows']:,}  \n**Trained:** {trace['model']['trained_at_utc']}  \n**Measurement period:** {trace['measurement_period']}")
    st.markdown("**Submitted inputs**")
    st.dataframe(pd.DataFrame([{"Input": key, "Value": value} for key, value in trace["input_values"].items()]), width="stretch", hide_index=True)
    st.markdown("**Synthetic Activity Reference components**")
    st.dataframe(pd.DataFrame(activity_baseline["components"])[[
        "source", "activity", "activity_unit", "emission_factor", "factor_unit", "emissions_kg_co2", "share_percent",
    ]], width="stretch", hide_index=True)
    st.markdown("**Factor provenance**")
    st.dataframe(pd.DataFrame(activity_baseline["components"])[[
        "source", "factor_id", "factor_source", "factor_geography", "factor_scope", "factor_version",
        "factor_effective_from", "factor_effective_to", "factor_verification_status", "factor_uncertainty",
        "registry_factor_value", "factor_methodology",
    ]], width="stretch", hide_index=True)
    st.caption(f"Factor registry: {activity_baseline['factor_registry']} ({activity_baseline['factor_registry_version']}). {activity_baseline['factor_registry_status']} Geography/scope: {activity_baseline['geography']} / {activity_baseline['scope']}. {trace['production_unit_note']}")
    st.markdown("**Derived model features**")
    st.dataframe(pd.DataFrame([{"Feature": key, "Value": value} for key, value in trace["model_features"].items()]), width="stretch", hide_index=True)
    st.markdown(
        f"""
1. **Synthetic Activity Reference:** submitted electricity, temperature, boiler pressure, and production are combined using the listed demo coefficients: **{activity_baseline['value']:,.1f} kg CO₂/hr**.
2. **ML prediction:** readings are transformed into electricity emissions (`kWh × submitted grid factor`), furnace temperature, boiler pressure, and production, then passed to the regression model: **{pred_val:,.1f} kg CO₂/hr**.
3. **Prediction interval:** ±{pred['prediction_interval']['half_width_kg_co2_per_hour']:.2f} kg CO₂/hr from chronological calibration residuals. Final-test observed coverage: {pred['prediction_interval']['observed_holdout_coverage']:.1%}.
4. **Difference from the synthetic reference:** {trace['difference_from_demo_baseline_kg_co2_per_hour']:+.2f} kg CO₂/hr. This is an ML-versus-demo comparison, not a measured adjustment.
"""
    )
    st.caption(pred["prediction_interval"]["caveat"] + " The baseline coefficients are synthetic and unverified; no official factor source, region, or reporting scope is configured.")

st.markdown("---")

st.markdown("## Synthetic Activity Reference Breakdown")
st.caption("These formula components are synthetic demonstration proxies, not measured source-wise facility emissions.")
st.dataframe(
    pd.DataFrame(activity_baseline["components"])[[
        "source", "activity", "activity_unit", "emission_factor", "factor_unit",
        "emissions_kg_co2", "share_percent", "factor_id", "factor_source",
        "factor_geography", "factor_scope", "factor_version", "factor_verification_status",
    ]],
    width="stretch",
    hide_index=True,
)
st.info("Future-horizon forecasting, anomaly detection, and equipment-level attribution are unavailable; no future schedule, configured detection baseline, or machine-level measurements are connected.")

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
colors = ["#a44c37" if v > 0 else "#3f6249" for v in vals]
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
    textfont=dict(size=11, color="#26332b"),
    hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.2f} kg CO₂<extra></extra>"
))
fig_shap.update_layout(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.55)",
    margin=dict(l=10, r=140, t=20, b=20),
    height=380,
    xaxis=dict(
        title="SHAP Value (kg CO₂ contribution)",
        gridcolor="#d8ded5", zeroline=True,
        zerolinecolor="#8e9a8e", color="#58645c"
    ),
    yaxis=dict(autorange="reversed", color="#26332b", tickfont=dict(size=11)),
    font=dict(family="IBM Plex Sans, sans-serif", color="#26332b"),
)
st.plotly_chart(fig_shap, width="stretch")

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

with st.expander("Global model behavior (fit-window contribution summary)"):
    st.caption("Mean absolute additive linear-model contributions over the fit window; these describe synthetic model behavior, not physical causation or real-factory importance.")
    global_importance = model.metrics.get("global_feature_importance", [])
    if global_importance:
        st.dataframe(pd.DataFrame(global_importance), width="stretch", hide_index=True)
    else:
        st.info("Global contribution metrics are unavailable for this model artifact.")

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

        run_sim = st.form_submit_button("⚡ Run What-If Simulation", width="stretch")

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
    sr4.metric("ML intensity, kg CO₂/submitted unit", f"{sim['baseline_intensity_kg_co2_per_production_unit']:.2f} → {sim['scenario_intensity_kg_co2_per_production_unit']:.2f}")

    st.caption(
        f"Separate Synthetic Activity Reference: {sim['baseline_activity_reference']['value']:,.1f} → "
        f"{sim['scenario_activity_reference']['value']:,.1f} kg CO₂/hr. These synthetic formula values are not verified accounting."
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

    baseline_intensity = sim["baseline_intensity_kg_co2_per_production_unit"]
    scenario_intensity = sim["scenario_intensity_kg_co2_per_production_unit"]
    if is_positive and scenario_intensity > baseline_intensity:
        st.warning("Absolute emissions decreased, but carbon intensity increased; production decreased faster than emissions in this model-estimated scenario.")
    elif reduction < 0 and scenario_intensity < baseline_intensity:
        st.info("Absolute emissions increased while carbon intensity decreased; the submitted production increase outpaced the emissions increase in this model estimate.")

    st.caption(f"Scenario model support: {sim['prediction_validity']['status']}. Results are model estimates, not guaranteed savings; verify after any approved intervention.")
    if sim["baseline_prediction_interval"]["available"] and sim["scenario_prediction_interval"]["available"]:
        base_interval = sim["baseline_prediction_interval"]
        scenario_interval = sim["scenario_prediction_interval"]
        st.caption(
            f"Empirical ranges (not confidence intervals): baseline {base_interval['lower_bound']:,.1f}–{base_interval['upper_bound']:,.1f} "
            f"kg CO₂/hr; scenario {scenario_interval['lower_bound']:,.1f}–{scenario_interval['upper_bound']:,.1f} kg CO₂/hr. "
            f"{scenario_interval['caveat']}"
        )
    production_change = sim["production_change"]
    st.caption(
        f"Submitted production change: {production_change['absolute_change_submitted_units_per_hour']:+.2f} units/hr "
        f"({production_change['percent_change'] if production_change['percent_change'] is not None else 'Unavailable'}%). "
        f"Cost impact: {sim['cost_impact']['reason']} Constraint status: {sim['scenario_constraints']['note']}"
    )
    with st.expander("Scenario inputs and calculation details"):
        scenario_rows = [
            {"Input": name, "Current": value, "Scenario": sim["scenario_inputs"].get(name, value)}
            for name, value in sim["baseline_inputs"].items()
        ]
        st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)
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
st.markdown("## Operations Review Prompts")
st.caption("Prototype screening rules use submitted readings and model contributions; they are not validated causal recommendations.")

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
                <div style="background:#ffffff;border:1px solid #ccd4ca;
                     border-radius:12px;padding:1.2rem;height:100%;margin-bottom:1rem;">
                    <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                         letter-spacing:.06em;color:#58645c;margin-bottom:.4rem;">{rec['category']}</div>
                    <div style="font-size:.98rem;font-weight:700;color:#26332b;margin-bottom:.5rem;">
                        {priority_icon} {rec['title']}
                    </div>
                    <div style="font-size:.82rem;color:#58645c;line-height:1.5;margin-bottom:.75rem;">
                        {rec['trigger']}<br/>{rec['reason']}<br/>{rec['impact_summary']}
                    </div>
                    <div style="font-size:.78rem;color:#3f6249;font-weight:600;">
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

st.markdown("## Historical Baseline, Targets & Verification")
st.info("Unavailable: no measured facility history, configured target, or post-intervention observations are connected. EcoSense does not display placeholder baselines or verification records.")

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
        "Weekend Flag": ("Yes" if is_weekend else "No") if is_weekend is not None else "Not supplied",
        "Previous Hour Energy (kWh)": energy_lag,
        "Previous Hour Production (Tons)": production_lag,
        "3-Hour Rolling Avg Energy (kWh)": rolling_avg_energy,
    }.items()])
    st.dataframe(echo_df, width="stretch", hide_index=True)

# ── Footer ─────────────────────────────────────────────────
footer_data_note = (
        "DEMO MODE: synthetic simulated inputs."
        if sample.get("demo_mode", False)
        else "Submitted inputs; source authenticity is not independently verified."
)
st.markdown("""
<div style='text-align:center;padding:1.5rem;color:#94a3b8;
      font-size:.78rem;border-top:1px solid rgba(255,255,255,.06);margin-top:1rem;'>
  EcoSense v3.0 &nbsp;·&nbsp; AI Carbon Optimization Copilot &nbsp;·&nbsp;
    Team EcoX &nbsp;·&nbsp; Physics-informed Regression + SHAP &nbsp;·&nbsp;
    {footer_data_note}
</div>
""".format(footer_data_note=footer_data_note), unsafe_allow_html=True)
