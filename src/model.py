import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Union
from src.accounting import calculate_activity_baseline
from src.validation import validate_operational_inputs, validate_scenario

class EcoSenseModel:
    """
    Core ML Engine for EcoSense: handles predictions, SHAP explainability,
    what-if simulations, and rule-based carbon reduction recommendations.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.elasticity = {}
        self.explainer = None
        self._linear_shap_background = None
        self._linear_shap_base_value = None
        self.feature_cols = []
        self.feature_names_friendly = {}
        
        self.load_artifacts()
        
    def load_artifacts(self):
        """Loads the trained model, evaluation metrics, and feature metadata."""
        model_path = os.path.join(self.models_dir, "ecosense_xgb.joblib")
        metrics_path = os.path.join(self.models_dir, "metrics.json")
        elasticity_path = os.path.join(self.models_dir, "elasticity.json")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        
        if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
            raise FileNotFoundError("Model artifacts not found. Please run 'python -m src.train_model' first.")
            
        self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        with open(metrics_path, "r") as f:
            self.metrics = json.load(f)
            
        with open(elasticity_path, "r") as f:
            self.elasticity = json.load(f)
            
        self.feature_cols = self.metrics.get("feature_cols", [])
        self.feature_names_friendly = self.metrics.get("feature_names_friendly", {})
        
        if hasattr(self.model, "coef_"):
            self._linear_shap_background = pd.DataFrame([{
                col: self.elasticity.get(col, {}).get("mean", 0.0)
                for col in self.feature_cols
            }])
            self._linear_shap_base_value = float(
                self.model.intercept_ + np.dot(self.model.coef_, self._linear_shap_background.iloc[0].to_numpy())
            )
        else:
            import shap

            self.explainer = shap.TreeExplainer(self.model)

    def prepare_df(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Ensures input data is formatted into a DataFrame matching feature columns."""
        if isinstance(data, dict):
            validate_operational_inputs(data)
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dict or pandas DataFrame")

        if "electricity_emissions_proxy" in self.feature_cols and "electricity_emissions_proxy" not in df.columns:
            if {"energy_kwh", "grid_emission_factor"}.issubset(df.columns):
                df["electricity_emissions_proxy"] = df["energy_kwh"] * df["grid_emission_factor"]
            else:
                df["electricity_emissions_proxy"] = self.elasticity.get("electricity_emissions_proxy", {}).get("mean", 0.0)
            
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = self.elasticity.get(col, {}).get("mean", 0.0)
                
        return df[self.feature_cols]

    def _feature_shap_values(self, features: pd.DataFrame):
        """Return background-relative SHAP values for linear or legacy tree estimators."""
        if self._linear_shap_background is not None:
            sample = features.iloc[0].to_numpy(dtype=float)
            background = self._linear_shap_background.iloc[0].to_numpy(dtype=float)
            contributions = (sample - background) * np.asarray(self.model.coef_, dtype=float)
            return self._linear_shap_base_value, contributions

        shap_values = self.explainer(features)
        base_value = float(np.asarray(self.explainer.expected_value).reshape(-1)[0])
        return base_value, shap_values.values[0]

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Predicts emissions with a holdout-calibrated residual range and traceable inputs.
        """
        X = self.prepare_df(data)
        preds = self.model.predict(X)
        rmse = self.metrics.get("test_rmse", 30.0)
        interval_half_width = self.metrics.get("prediction_interval_abs_error_q95")
        interval_coverage = self.metrics.get("prediction_interval_holdout_coverage")
        
        results = []
        for p in preds:
            pred_val = float(p)
            lower_bound = max(0.0, pred_val - interval_half_width) if interval_half_width is not None else None
            upper_bound = pred_val + interval_half_width if interval_half_width is not None else None
            results.append({
                "predicted_emissions_kg_co2": round(pred_val, 2),
                "unit": "kg CO2/hour",
                "method": "ML prediction",
                "lower_bound_95": round(lower_bound, 2) if lower_bound is not None else None,
                "upper_bound_95": round(upper_bound, 2) if upper_bound is not None else None,
                "test_r2": round(self.metrics.get("test_r2", 0.99), 4),
                "test_rmse": round(rmse, 2),
                "prediction_interval": {
                    "method": self.metrics.get("prediction_interval_method", "Unavailable"),
                    "nominal_coverage": self.metrics.get("prediction_interval_nominal_coverage"),
                    "observed_holdout_coverage": interval_coverage,
                    "half_width_kg_co2_per_hour": round(interval_half_width, 2) if interval_half_width is not None else None,
                    "caveat": "Holdout residual range; future coverage is not guaranteed, especially under time or operating-regime shift." if interval_half_width is not None else "Prediction interval unavailable for this model version."
                }
            })

        if not isinstance(data, dict):
            return {"predictions": results}

        result = results[0]
        baseline = calculate_activity_baseline(data)
        explanation = self.explain(data)
        production = float(data["production_volume_tons"])
        training_ranges = self.metrics.get("training_input_ranges", {})
        out_of_training_range = [
            key for key, bounds in training_ranges.items()
            if float(data[key]) < bounds["min"] or float(data[key]) > bounds["max"]
        ]
        support_status = "review: input outside model-fit range" if out_of_training_range else "within observed model-fit ranges"
        result.update({
            "traceability": {
                "input_values": {key: value for key, value in data.items()},
                "model_features": {key: float(value) for key, value in X.iloc[0].to_dict().items()},
                "activity_based_demo_baseline": baseline,
                "difference_from_demo_baseline_kg_co2_per_hour": round(pred_val - baseline["value"], 2),
                "model": {
                    "name": self.metrics.get("model_type", type(self.model).__name__),
                    "version": self.metrics.get("model_version", "Not versioned"),
                    "feature_schema_version": self.metrics.get("feature_schema_version", "Not versioned"),
                    "training_rows": self.metrics.get("model_training_rows"),
                    "dataset_rows": self.metrics.get("n_samples"),
                    "model_feature_count": len(self.feature_cols),
                    "calibration_rows": self.metrics.get("calibration_rows"),
                    "test_rows": self.metrics.get("test_rows"),
                    "calibration_start_timestamp": self.metrics.get("calibration_start_timestamp"),
                    "test_start_timestamp": self.metrics.get("test_start_timestamp"),
                    "training_data_sha256": self.metrics.get("training_data_sha256"),
                    "trained_at_utc": self.metrics.get("trained_at_utc"),
                    "evaluation_method": self.metrics.get("evaluation_method", "Not recorded"),
                    "test_mae_kg_co2_per_hour": self.metrics.get("test_mae"),
                    "test_rmse_kg_co2_per_hour": self.metrics.get("test_rmse"),
                    "test_r2": self.metrics.get("test_r2"),
                },
                "input_quality": {
                    "status": "All required demo fields supplied; configured bounds passed",
                    "completeness_percent": 100.0,
                    "required_input_count": 11,
                    "missing_inputs": [],
                    "range_validation_status": "passed",
                    "source_verification": "not available",
                    "training_range_status": support_status,
                    "outside_training_range": out_of_training_range,
                    "limitations": "Completeness is field presence only. Configured demo bounds are not equipment safety limits; sensor freshness, anomaly, and source authenticity are not checked."
                },
                "measurement_period": f"One-hour operating interval at hour {data['hour']:02d}; measurement date was not submitted.",
                "production_unit_note": "Production intensity uses the submitted source unit named 'tons'; metric versus short ton is unspecified and no conversion is applied.",
                "feature_contributions": explanation["feature_contributions"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "carbon_intensity_kg_co2_per_production_unit": round(pred_val / production, 3) if production > 0 else None,
        })
        return result

    def explain(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes SHAP waterfall explanation decomposing the prediction into feature contributions.
        Returns φ₀ (baseline average), φᵢ (feature SHAP values), and percentage impacts.
        """
        X = self.prepare_df(data)
        base_value, sample_shap = self._feature_shap_values(X)
        feature_values = X.iloc[0].to_dict()
        
        predicted_emission = base_value + np.sum(sample_shap)
        total_abs_shap = np.sum(np.abs(sample_shap)) + 1e-9
        
        breakdown = []
        for col, val_shap in zip(self.feature_cols, sample_shap):
            raw_val = float(feature_values[col])
            shap_val = float(val_shap)
            pct = (abs(shap_val) / total_abs_shap) * 100.0
            
            breakdown.append({
                "feature": col,
                "friendly_name": self.feature_names_friendly.get(col, col),
                "feature_value": round(raw_val, 2),
                "shap_value_kg_co2": round(shap_val, 2),
                "impact_percentage": round(pct, 1),
                "direction": "INCREASE" if shap_val > 0 else "DECREASE"
            })
            
        # Sort features by absolute SHAP impact
        breakdown.sort(key=lambda x: abs(x["shap_value_kg_co2"]), reverse=True)
        
        return {
            "method": "SHAP model feature contribution",
            "causality_note": "Feature contributions explain model behavior and do not establish physical causation.",
            "baseline_emission_kg_co2": round(base_value, 2),
            "predicted_emission_kg_co2": round(float(predicted_emission), 2),
            "feature_contributions": breakdown
        }

    def simulate(self, baseline_data: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates hypothetical operational changes (what-if scenarios).
        Compares model predictions and attributes the change with SHAP values.
        """
        validate_operational_inputs(baseline_data)
        scenario_data = validate_scenario(baseline_data, modifications)
        X_base = self.prepare_df(baseline_data)
        base_pred = float(self.model.predict(X_base)[0])
        X_sim = self.prepare_df(scenario_data)
            
        sim_pred = float(self.model.predict(X_sim)[0])
        
        delta_kg = round(base_pred - sim_pred, 2)
        pct_change = round((delta_kg / (base_pred + 1e-9)) * 100.0, 2)
        baseline_production = float(baseline_data["production_volume_tons"])
        scenario_production = float(scenario_data["production_volume_tons"])
        
        _, base_shap = self._feature_shap_values(X_base)
        _, sim_shap = self._feature_shap_values(X_sim)
        feature_savings = []
        for index, col in enumerate(self.feature_cols):
            original_value = float(X_base[col].iloc[0])
            simulated_value = float(X_sim[col].iloc[0])
            attribution = float(base_shap[index] - sim_shap[index])
            feature_savings.append({
                "feature": col,
                "friendly_name": self.feature_names_friendly.get(col, col),
                "original_value": round(original_value, 2),
                "simulated_value": round(simulated_value, 2),
                "model_contribution_change_kg_co2_per_hour": round(-attribution, 2),
                "estimated_saving_kg_co2": round(attribution, 2)
            })
        feature_savings.sort(key=lambda item: abs(item["estimated_saving_kg_co2"]), reverse=True)
                
        return {
            "method": "Model-estimated scenario",
            "unit": "kg CO2/hour",
            "limitation": "Estimated change only; actual reduction must be verified using post-intervention measurements.",
            "baseline_activity_based_demo": calculate_activity_baseline(baseline_data),
            "scenario_activity_based_demo": calculate_activity_baseline(scenario_data),
            "baseline_emission_kg_co2": round(base_pred, 2),
            "simulated_emission_kg_co2": round(sim_pred, 2),
            "baseline_intensity_kg_co2_per_production_unit": round(base_pred / baseline_production, 3) if baseline_production > 0 else None,
            "scenario_intensity_kg_co2_per_production_unit": round(sim_pred / scenario_production, 3) if scenario_production > 0 else None,
            "baseline_inputs": baseline_data,
            "scenario_inputs": scenario_data,
            "carbon_reduction_kg_co2": delta_kg,
            "percentage_reduction": pct_change,
            "is_net_positive_reduction": delta_kg > 0,
            "parameter_impacts": feature_savings,
            "parameter_impact_semantics": {
                "model_contribution_change_kg_co2_per_hour": "Scenario feature contribution minus current feature contribution; positive means higher modeled emissions.",
                "estimated_saving_kg_co2": "Deprecated compatibility alias; positive means lower model estimate, not verified savings."
            }
        }

    def generate_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes operational inputs and SHAP explanation to output templated carbon optimization recommendations.
        """
        explanation = self.explain(data)
        contributions = {item["feature"]: item for item in explanation["feature_contributions"]}

        recs = []
        
        # Rule 1: High Grid Emission Factor
        grid_factor = data.get("grid_emission_factor", 0.6)
        if grid_factor > 0.55:
            recs.append({
                "id": "REC-01",
                "title": "Review electricity sourcing and load timing",
                "category": "Electricity review",
                "priority": "HIGH",
                "trigger": f"Submitted grid factor {grid_factor:.3f} kg CO2/kWh exceeds the demo screening threshold 0.55.",
                "reason": f"The model's electricity emissions feature contribution is {contributions['electricity_emissions_proxy']['shap_value_kg_co2']:+.1f} kg CO2/hour.",
                "potential_saving_kg_co2_day": None,
                "impact_summary": "No saving estimate is available; check the factor source and evaluate an operator-approved scenario.",
                "cost_impact": "Unavailable; no tariff data configured.",
                "production_impact": "Unavailable; production response is not validated.",
                "confidence": None,
                "constraint_note": "Verify the factor source and preserve site operating constraints.",
                "ranking_reason": "Triggered by a demo screening rule; no verified impact ranking is available.",
                "action_steps": [
                    "Check the factor against a dated utility/provider source for the correct region.",
                    "Review time-specific electricity options actually available to the site.",
                    "Compare candidate changes in the simulator and validate with measured data."
                ],
                "implementation_effort": "Medium"
            })
            
        # Rule 2: Furnace Temperature Optimization
        temp = data.get("furnace_temp_c", 1000)
        if temp > 980:
            recs.append({
                "id": "REC-02",
                "title": "Review furnace thermal performance",
                "category": "Thermal process review",
                "priority": "HIGH" if temp > 1050 else "MEDIUM",
                "trigger": f"Furnace reading {temp:.1f} degrees C exceeds the demo screening threshold 980 degrees C; this is not a process limit.",
                "reason": f"The model's furnace temperature feature contribution is {contributions['furnace_temp_c']['shap_value_kg_co2']:+.1f} kg CO2/hour.",
                "potential_saving_kg_co2_day": None,
                "impact_summary": "No saving estimate is available; review the reading with an operator before testing changes.",
                "cost_impact": "Unavailable; no energy tariff configured.",
                "production_impact": "Unavailable; thermal effects on production are not validated.",
                "confidence": None,
                "constraint_note": "Do not exceed product and equipment process limits.",
                "ranking_reason": "Triggered by a demo screening rule; no verified impact ranking is available.",
                "action_steps": [
                    "Compare the reading with the control system and calibration record.",
                    "Review insulation and settings with a qualified process engineer.",
                    "Only simulate alternatives allowed by the approved process window."
                ],
                "implementation_effort": "Low"
            })

        # Rule 3: Energy Spikes vs Production Ratio
        energy = data.get("energy_kwh", 2000)
        prod = data.get("production_volume_tons", 40)
        specific_energy = energy / (prod + 1e-5) # kWh / ton
        
        if specific_energy is not None and specific_energy > 55.0:
            recs.append({
                "id": "REC-03",
                "title": "Review energy use per tonne of product",
                "category": "Energy intensity review",
                "priority": "MEDIUM",
                "trigger": f"Calculated intensity is {specific_energy:.1f} kWh/tonne, above the demo screening threshold 55; no industry benchmark is configured.",
                "reason": "Energy use normalized by submitted production triggered a prototype screening rule.",
                "potential_saving_kg_co2_day": None,
                "impact_summary": "No saving estimate is available; review aligned energy and production records before testing changes.",
                "cost_impact": "Unavailable; no energy tariff configured.",
                "production_impact": "Unavailable; no validated production-response model configured.",
                "confidence": None,
                "constraint_note": "Preserve required production output and product-quality limits.",
                "ranking_reason": "Triggered by a demo screening rule; no verified impact ranking is available.",
                "action_steps": [
                    "Compare energy and production meters over the same time interval.",
                    "Review equipment state, operating schedule, and maintenance history.",
                    "Simulate only a specific operator-approved alternative."
                ],
                "implementation_effort": "Low"
            })
            
        # Default fallback recommendation
        if not recs:
            recs.append({
                "id": "REC-04",
                "title": "No demo screening rule was triggered",
                "category": "Screening status",
                "priority": "LOW",
                "trigger": "Submitted values did not exceed prototype heuristic review thresholds.",
                "reason": "This does not establish that the facility is operating optimally.",
                "potential_saving_kg_co2_day": None,
                "impact_summary": "No impact estimate is available.",
                "cost_impact": "Unavailable.",
                "production_impact": "Unavailable.",
                "confidence": None,
                "constraint_note": "Use measured site data and operator judgment.",
                "ranking_reason": "No quantified recommendation is available for ranking.",
                "action_steps": [
                    "Maintain weekly sub-metering calibration.",
                    "Review ISO 50001 energy audit checklists."
                ],
                "implementation_effort": "Low"
            })
            
        return recs
