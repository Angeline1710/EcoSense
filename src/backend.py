import os
import json
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.model import EcoSenseModel
from src.validation import validate_operational_inputs

app = FastAPI(
    title="EcoSense API — AI Carbon Optimization Copilot",
    description="Synthetic-data industrial emissions prototype with traceable model predictions, SHAP feature contributions, and scenario estimates.",
    version="3.0.0"
)

# CORS middleware for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates if present
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
static_dir = os.path.join(frontend_dir, "static")
templates_dir = os.path.join(frontend_dir, "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Instantiate Core ML Model Engine
try:
    ecosense = EcoSenseModel(models_dir="models")
except Exception as e:
    print(f"[Backend Warning] Could not load model immediately: {e}")
    ecosense = None

# Pydantic Schemas
class OperationalInput(BaseModel):
    energy_kwh: float = Field(ge=0, le=10000, description="Measured or manually supplied energy, kWh")
    grid_emission_factor: float = Field(ge=0, le=2, description="User-supplied grid factor, kg CO2/kWh; provenance is not independently verified")
    production_volume_tons: float = Field(ge=0, le=500, description="Production output, tonnes/hour")
    furnace_temp_c: float = Field(ge=0, le=2000, description="Furnace temperature, degrees C")
    boiler_pressure_bar: float = Field(ge=0, le=100, description="Boiler pressure, bar")
    shift: int | None = Field(default=None, ge=1, le=3, description="Optional shift context; not used by the active model")
    hour: int | None = Field(default=None, ge=0, le=23, description="Optional local hour context; not used by the active model")
    day_of_week: int | None = Field(default=None, ge=0, le=6, description="Optional day context; not used by the active model")
    is_weekend: int | None = Field(default=None, ge=0, le=1, description="Optional weekend context; not used by the active model")
    energy_lag1: float | None = Field(default=None, ge=0, le=10000, description="Optional experimental lag; not used by the active model")
    production_lag1: float | None = Field(default=None, ge=0, le=500, description="Optional experimental lag; not used by the active model")
    rolling_avg_energy_3h: float | None = Field(default=None, ge=0, le=10000, description="Optional experimental rolling feature; not used by the active model")

    @model_validator(mode="after")
    def validate_operating_context(self):
        validate_operational_inputs(self.model_dump())
        return self

class SimulationInput(BaseModel):
    baseline_data: OperationalInput
    modifications: Dict[str, float] = Field(..., description="Explicitly supplied scenario changes; no scenario values are assumed")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serves the EcoSense overview page."""
    home_path = os.path.join(templates_dir, "home.html")
    if os.path.exists(home_path):
        with open(home_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>EcoSense</h1><p>Visit <a href='/dashboard'>the dashboard</a> or <a href='/docs'>API documentation</a>.</p>")

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the interactive emissions analysis dashboard."""
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>EcoSense Backend Service Operational</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")

@app.get("/api/status")
async def get_status():
    """Returns model health, accuracy metrics (R2, RMSE), and feature schema."""
    if not ecosense:
        raise HTTPException(status_code=500, detail="EcoSense ML model engine not initialized.")
    return {
        "status": "ONLINE",
        "model_type": ecosense.metrics.get("model_type", type(ecosense.model).__name__),
        "metrics": ecosense.metrics,
        "feature_count": ecosense.metrics.get("input_feature_count", len(ecosense.feature_cols)),
        "input_feature_count": ecosense.metrics.get("input_feature_count", len(ecosense.feature_cols)),
        "model_feature_count": len(ecosense.feature_cols),
    }

@app.get("/api/data/recent")
async def get_recent_data(limit: int = 48):
    """Returns recent historical time series dataset records for dashboard plotting."""
    if limit < 1 or limit > 2160:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 2160.")
    csv_path = "data/industrial_emissions.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=444, detail="Dataset file missing.")
    df = pd.read_csv(csv_path)
    records = df.tail(limit).to_dict(orient="records")
    return {
        "total_records": len(df),
        "limit": limit,
        "data_mode": "Synthetic demonstration data",
        "emissions_method": "Generator formula plus synthetic noise; not measured/accounted facility emissions",
        "unit": "kg CO2/hour",
        "data": records,
    }

@app.post("/api/predict")
async def predict_emissions(payload: OperationalInput):
    """Returns an ML estimate, empirical residual range, input trace, and demo baseline."""
    if not ecosense:
        raise HTTPException(status_code=500, detail="Model engine offline.")
    data_dict = payload.model_dump(exclude_none=True)
    result = ecosense.predict(data_dict)
    return result

@app.post("/api/explain")
async def explain_prediction(payload: OperationalInput):
    """Computes SHAP waterfall feature contribution breakdown for input sample."""
    if not ecosense:
        raise HTTPException(status_code=500, detail="Model engine offline.")
    data_dict = payload.model_dump(exclude_none=True)
    explanation = ecosense.explain(data_dict)
    return explanation

@app.post("/api/simulate")
async def simulate_scenario(payload: SimulationInput):
    """Simulates what-if scenarios and calculates net carbon reduction and savings breakdown."""
    if not ecosense:
        raise HTTPException(status_code=500, detail="Model engine offline.")
    base_dict = payload.baseline_data.model_dump(exclude_none=True)
    try:
        sim_result = ecosense.simulate(base_dict, payload.modifications)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    sim_result["baseline_inputs"] = base_dict
    sim_result["scenario_inputs"] = {**base_dict, **payload.modifications}
    return sim_result

@app.post("/api/recommend")
async def get_recommendations(payload: OperationalInput):
    """Returns dynamic carbon optimization recommendations based on operational parameters and SHAP analysis."""
    if not ecosense:
        raise HTTPException(status_code=500, detail="Model engine offline.")
    data_dict = payload.model_dump(exclude_none=True)
    prediction = ecosense.predict(data_dict)
    recs = ecosense.generate_recommendations(data_dict) if prediction["prediction_validity"]["prediction_available"] else []
    return {
        "recommendations": recs,
        "count": len(recs),
        "model_version": ecosense.metrics.get("model_version"),
        "prediction_validity": prediction["prediction_validity"],
        "model_support": prediction["model_support"],
    }

@app.get("/api/report")
async def generate_esg_report():
    """Generates a transparent summary of synthetic demo labels and model evaluation."""
    csv_path = "data/industrial_emissions.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Dataset missing")
        
    df = pd.read_csv(csv_path)
    avg_daily_emissions = round(float(df["emissions_kg_co2"].mean()) * 24 / 1000, 2)
    total_emissions = round(float(df["emissions_kg_co2"].sum()) / 1000, 2)
    avg_grid_factor = round(float(df["grid_emission_factor"].mean()), 3)
    synthetic_label_intensity = round(float(df["emissions_kg_co2"].sum() / df["production_volume_tons"].sum()), 3)
    model_metadata = ecosense.metrics if ecosense else {}
    missing_values = {column: int(count) for column, count in df.isna().sum().items()}
    total_cells = int(df.shape[0] * df.shape[1])
    field_completeness = round(100.0 * (1.0 - sum(missing_values.values()) / total_cells), 2) if total_cells else None
    limitations = [
        "Bundled hourly records and targets are synthetic.",
        "Grid factor geography and source are not verified.",
        "Thermal and production coefficients are generator assumptions, not authoritative emission factors.",
        "No sensor freshness, drift, or post-intervention verification is available.",
    ]
    
    return {
        "report_title": "EcoSense Synthetic Demonstration Summary",
        "data_classification": "Synthetic demonstration labels; not measured or verified facility emissions",
        "timeframe": f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}" if "timestamp" in df else f"{len(df)} hourly samples",
        "summary": {
            "synthetic_label_total_mt_co2": total_emissions,
            "synthetic_label_average_mt_co2_per_day": avg_daily_emissions,
            "synthetic_grid_factor_average_kg_co2_per_kwh": avg_grid_factor,
            "synthetic_label_intensity_kg_co2_per_production_unit": synthetic_label_intensity,
            "accounted_facility_emissions": None,
            "measured_post_intervention_emissions": None,
            "model_test_r2": model_metadata.get("test_r2"),
            "model_test_rmse_kg_co2_per_hour": model_metadata.get("test_rmse"),
            "model_test_mae_kg_co2_per_hour": model_metadata.get("test_mae"),
            "model_version": model_metadata.get("model_version"),
            "test_period_start": model_metadata.get("test_start_timestamp"),
        },
        "emissions_taxonomy": {
            "accounted": "Unavailable; no authoritative factors or verified activity data configured",
            "ml_prediction": "Available per submitted request; see /api/predict",
            "scenario_estimate": "Available per scenario request; not guaranteed",
            "verified_post_intervention": "Unavailable; no implementation verification records exist",
        },
        "sections": {
            "accounted_emissions": {
                "status": "UNAVAILABLE",
                "reason": "No authoritative, site-configured factors or verified activity records are available.",
            },
            "ml_estimates": {
                "status": "PER_REQUEST_ONLY",
                "description": "Current-state estimates are returned by /api/predict; no aggregate facility estimate is stored in this report.",
            },
            "scenario_estimates": {
                "status": "PER_REQUEST_ONLY",
                "description": "Explicit what-if estimates are returned by /api/simulate and are not guaranteed savings.",
            },
            "observed_verified_emissions": {
                "status": "UNAVAILABLE",
                "value": None,
                "reason": "No measured post-intervention observations are recorded.",
            },
            "carbon_intensity": {
                "status": "SYNTHETIC_LABEL_ONLY",
                "value_kg_co2_per_submitted_production_unit": synthetic_label_intensity,
                "production_unit_note": "The source unit named tons is not documented as metric or short tons; no conversion is applied.",
            },
            "data_quality": {
                "classification": "SYNTHETIC_DEMONSTRATION_DATA",
                "field_completeness_percent": field_completeness,
                "missing_values_by_column": missing_values,
                "source_authenticity": "NOT_VERIFIED",
                "sensor_freshness": "NOT_ASSESSED",
                "anomalies": "NOT_ASSESSED",
            },
            "model_information": {
                "model_type": model_metadata.get("model_type"),
                "model_version": model_metadata.get("model_version"),
                "features": model_metadata.get("feature_cols"),
                "training_rows": model_metadata.get("model_training_rows"),
                "calibration_rows": model_metadata.get("calibration_rows"),
                "test_rows": model_metadata.get("test_rows"),
                "mae_kg_co2_per_hour": model_metadata.get("test_mae"),
                "rmse_kg_co2_per_hour": model_metadata.get("test_rmse"),
                "r2": model_metadata.get("test_r2"),
                "prediction_interval_method": model_metadata.get("prediction_interval_method"),
                "training_environment": model_metadata.get("model_environment"),
            },
            "major_drivers": "Available per prediction through /api/explain; no aggregate drivers are claimed in this report.",
            "recommendations": "Screening prompts are available per request; no verified financial or production impact is configured.",
            "verification": "Unavailable; implementation and post-intervention measurement records do not exist.",
            "limitations": limitations,
        },
        "regulatory_compliance": {"status": "NOT ASSESSED", "reason": "This prototype does not implement a regulatory accounting methodology or independent audit."},
        "limitations": limitations,
    }
