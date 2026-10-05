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
    shift: int = Field(ge=1, le=3, description="Shift number matching the submitted hour")
    hour: int = Field(ge=0, le=23, description="Local hour of day")
    day_of_week: int = Field(ge=0, le=6, description="Day of week, Monday=0 through Sunday=6")
    is_weekend: int = Field(ge=0, le=1, description="Weekend flag derived from day of week")
    energy_lag1: float = Field(ge=0, le=10000, description="Previous hour energy, kWh")
    production_lag1: float = Field(ge=0, le=500, description="Previous hour production, tonnes")
    rolling_avg_energy_3h: float = Field(ge=0, le=10000, description="Current and prior two-hour mean energy, kWh")

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
    data_dict = payload.model_dump()
    result = ecosense.predict(data_dict)
    return result

@app.post("/api/explain")
async def explain_prediction(payload: OperationalInput):
    """Computes SHAP waterfall feature contribution breakdown for input sample."""
    if not ecosense:
        raise HTTPException(status_code=500, detail="Model engine offline.")
    data_dict = payload.model_dump()
    explanation = ecosense.explain(data_dict)
    return explanation

@app.post("/api/simulate")
async def simulate_scenario(payload: SimulationInput):
    """Simulates what-if scenarios and calculates net carbon reduction and savings breakdown."""
    if not ecosense:
        raise HTTPException(status_code=500, detail="Model engine offline.")
    base_dict = payload.baseline_data.model_dump()
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
    data_dict = payload.model_dump()
    recs = ecosense.generate_recommendations(data_dict)
    return {"recommendations": recs, "count": len(recs)}

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
        "regulatory_compliance": {"status": "NOT ASSESSED", "reason": "This prototype does not implement a regulatory accounting methodology or independent audit."},
        "limitations": [
            "Bundled hourly records and targets are synthetic.",
            "Grid factor geography and source are not verified.",
            "Thermal and production coefficients are generator assumptions, not authoritative emission factors.",
            "No sensor freshness, drift, or post-intervention verification is available.",
        ],
    }
