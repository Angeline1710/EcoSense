import os
import json
import joblib
import numpy as np
import pandas as pd
from importlib.metadata import version
from datetime import datetime, timezone
from typing import Dict, Any, List, Union
from src.accounting import calculate_activity_baseline
from src.validation import REQUIRED_INPUTS, validate_operational_inputs, validate_scenario

class EcoSenseModel:
    """
    Core ML Engine for EcoSense: handles predictions, SHAP explainability,
    what-if simulations, and rule-based carbon reduction recommendations.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
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
        environment_path = os.path.join(self.models_dir, "model_environment.json")
        
        if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
            raise FileNotFoundError("Model artifacts not found. Please run 'python -m src.train_model' first.")

        if not os.path.exists(environment_path):
            raise FileNotFoundError("Model environment manifest not found. Retrain the model before inference.")

        with open(environment_path, "r", encoding="utf-8") as environment_file:
            self.model_environment = json.load(environment_file)

        runtime_versions = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": version("scikit-learn"),
            "scipy": version("scipy"),
            "joblib": joblib.__version__,
        }
        incompatible = {
            package: {"artifact": expected, "runtime": runtime_versions[package]}
            for package, expected in self.model_environment.items()
            if package in runtime_versions and expected != runtime_versions[package]
        }
        if incompatible:
            raise RuntimeError(f"Model environment mismatch; expected pinned versions: {incompatible}")
            
        self.model = joblib.load(model_path)
            
        with open(metrics_path, "r") as f:
            self.metrics = json.load(f)
            
        with open(elasticity_path, "r") as f:
            self.elasticity = json.load(f)
            
        self.feature_cols = self.metrics.get("feature_cols", [])
        self.feature_names_friendly = self.metrics.get("feature_names_friendly", {})
        
        if hasattr(self.model, "coef_"):
            missing_explanation_metadata = [
                col for col in self.feature_cols
                if col not in self.elasticity or "mean" not in self.elasticity[col]
            ]
            if missing_explanation_metadata:
                raise RuntimeError(
                    "Missing trained explanation baseline for model feature(s): "
                    + ", ".join(missing_explanation_metadata)
                )
            self._linear_shap_background = pd.DataFrame([{
                col: self.elasticity[col]["mean"]
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
                raise ValueError("Missing required inputs: energy_kwh and grid_emission_factor.")
            
        for col in self.feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required model feature: {col}.")
                
        return df[self.feature_cols]

    def _assess_model_support(self, data: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
        """Compare raw inputs and engineered features with observed fit-window ranges."""
        ranges = dict(self.metrics.get("training_input_ranges", {}))
        model_feature_ranges = self.metrics.get("model_feature_ranges", {})
        for feature in self.feature_cols:
            stats = model_feature_ranges.get(feature, self.elasticity.get(feature, {}))
            if "min" in stats and "max" in stats:
                ranges[feature] = {"min": stats["min"], "max": stats["max"]}

        checks = []
        for name, bounds in ranges.items():
            if name in data:
                value = float(data[name])
            elif name in features.columns:
                value = float(features.iloc[0][name])
            else:
                continue

            minimum = float(bounds["min"])
            maximum = float(bounds["max"])
            distance = max(minimum - value, value - maximum, 0.0)
            span = maximum - minimum
            distance_in_training_spans = distance / span if span > 0 else (0.0 if distance == 0 else float("inf"))
            checks.append({
                "feature": name,
                "current_value": value,
                "training_min": minimum,
                "training_max": maximum,
                "within_range": distance == 0,
                "distance_from_range": distance,
                "distance_in_training_spans": distance_in_training_spans,
            })

        affected = [check for check in checks if not check["within_range"]]
        if any(check["distance_in_training_spans"] > 1.0 for check in affected):
            status = "UNSUPPORTED"
        elif affected:
            status = "DEGRADED"
        else:
            status = "SUPPORTED"

        return {
            "status": status,
            "within_range": not affected and bool(checks),
            "distance_from_range": {check["feature"]: check["distance_from_range"] for check in affected},
            "affected_features": [check["feature"] for check in affected],
            "feature_ranges": checks,
            "unsupported_threshold": "More than one observed training-range span beyond a feature boundary.",
        }

    @staticmethod
    def _prediction_validity(raw_prediction: float, model_support: Dict[str, Any]) -> Dict[str, Any]:
        if not np.isfinite(raw_prediction) or raw_prediction < 0:
            return {
                "status": "UNSUPPORTED",
                "prediction_available": False,
                "reason": "The model output is non-finite or negative; negative emissions are physically invalid.",
            }
        if model_support["status"] == "UNSUPPORTED":
            return {
                "status": "UNSUPPORTED",
                "prediction_available": False,
                "reason": "Input conditions exceed the configured extrapolation limit for the observed training domain.",
            }
        if model_support["status"] == "DEGRADED":
            return {
                "status": "DEGRADED",
                "prediction_available": True,
                "reason": "At least one input is outside observed model-fit ranges; interpret this estimate cautiously.",
            }
        return {
            "status": "VALID",
            "prediction_available": True,
            "reason": "All evaluated inputs are within observed model-fit ranges and the estimate is non-negative.",
        }

    def _empirical_prediction_interval(self, prediction: float) -> Dict[str, Any]:
        half_width = self.metrics.get("prediction_interval_abs_error_q95")
        if half_width is None or not np.isfinite(float(half_width)) or float(half_width) < 0:
            return {"available": False, "lower_bound": None, "upper_bound": None, "method": None}
        half_width = float(half_width)
        return {
            "available": True,
            "lower_bound": round(max(0.0, prediction - half_width), 2),
            "upper_bound": round(prediction + half_width, 2),
            "nominal_coverage": self.metrics.get("prediction_interval_nominal_coverage"),
            "observed_holdout_coverage": self.metrics.get("prediction_interval_holdout_coverage"),
            "half_width_kg_co2_per_hour": round(half_width, 2),
            "method": self.metrics.get("prediction_interval_method"),
            "caveat": "Empirical residual range calibrated on synthetic data; future coverage is not guaranteed. Lower bound is truncated at zero.",
        }

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
        Estimate emissions only when the model output is physically and statistically supported.
        """
        X = self.prepare_df(data)
        raw_predictions = np.asarray(self.model.predict(X), dtype=float)
        interval_half_width = self.metrics.get("prediction_interval_abs_error_q95")
        interval_coverage = self.metrics.get("prediction_interval_holdout_coverage")
        if interval_half_width is not None:
            interval_half_width = float(interval_half_width)
            if not np.isfinite(interval_half_width) or interval_half_width < 0:
                interval_half_width = None

        results = []
        for index, raw_prediction in enumerate(raw_predictions):
            row = data.iloc[index].to_dict() if isinstance(data, pd.DataFrame) else data
            row_features = X.iloc[[index]]
            model_support = self._assess_model_support(row, row_features)
            validity = self._prediction_validity(float(raw_prediction), model_support)
            available = validity["prediction_available"]
            predicted_value = round(float(raw_prediction), 2) if available else None
            lower_bound = max(0.0, float(raw_prediction) - interval_half_width) if available and interval_half_width is not None else None
            upper_bound = float(raw_prediction) + interval_half_width if available and interval_half_width is not None else None
            results.append({
                "predicted_emissions_kg_co2": predicted_value,
                "unit": "kg CO2/hour",
                "method": "Current-state ML estimation" if available else "Prediction unavailable",
                "prediction_validity": validity,
                "model_support": model_support,
                "lower_bound_95": round(lower_bound, 2) if lower_bound is not None else None,
                "upper_bound_95": round(upper_bound, 2) if upper_bound is not None else None,
                "test_r2": self.metrics.get("test_r2"),
                "test_rmse": self.metrics.get("test_rmse"),
                "prediction_interval": {
                    "available": available and interval_half_width is not None,
                    "method": self.metrics.get("prediction_interval_method") if interval_half_width is not None else None,
                    "nominal_coverage": self.metrics.get("prediction_interval_nominal_coverage") if interval_half_width is not None else None,
                    "observed_holdout_coverage": interval_coverage if interval_half_width is not None else None,
                    "half_width_kg_co2_per_hour": round(interval_half_width, 2) if interval_half_width is not None else None,
                    "lower_bound_policy": "Truncated at the physical zero-emissions boundary." if available and interval_half_width is not None else None,
                    "caveat": "Empirical residual interval calibrated on synthetic data; observed holdout coverage is not a future guarantee." if available and interval_half_width is not None else "Prediction interval unavailable for this result.",
                },
            })

        if not isinstance(data, dict):
            return {"predictions": results}

        result = results[0]
        baseline = calculate_activity_baseline(data)
        explanation = self.explain(data) if result["prediction_validity"]["prediction_available"] else None
        production = float(data["production_volume_tons"])
        support_status = result["model_support"]["status"]
        result.update({
            "emissions_classification": "ML-ESTIMATED",
            "model_version": self.metrics.get("model_version"),
            "data_quality": {
                "status": "REQUIRED_INPUTS_COMPLETE",
                "completeness_percent": 100.0,
                "missing_inputs": [],
                "assessment_scope": "Required field presence and configured numeric bounds only.",
                "not_assessed": ["source authenticity", "sensor freshness", "timestamp consistency", "sensor anomalies"],
            },
            "demo_mode": data.get("demo_mode"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "training_range": result["model_support"]["feature_ranges"],
            "top_drivers": explanation["feature_contributions"][:3] if explanation else [],
            "traceability": {
                "input_values": {key: value for key, value in data.items()},
                "model_features": {key: float(value) for key, value in X.iloc[0].to_dict().items()},
                "activity_reference": baseline,
                "activity_based_demo_baseline": baseline,
                "difference_from_demo_baseline_kg_co2_per_hour": round(result["predicted_emissions_kg_co2"] - baseline["value"], 2) if result["predicted_emissions_kg_co2"] is not None else None,
                "model": {
                    "name": self.metrics.get("model_type", type(self.model).__name__),
                    "version": self.metrics.get("model_version"),
                    "feature_schema_version": self.metrics.get("feature_schema_version"),
                    "training_rows": self.metrics.get("model_training_rows"),
                    "dataset_rows": self.metrics.get("n_samples"),
                    "model_feature_count": len(self.feature_cols),
                    "calibration_rows": self.metrics.get("calibration_rows"),
                    "test_rows": self.metrics.get("test_rows"),
                    "calibration_start_timestamp": self.metrics.get("calibration_start_timestamp"),
                    "test_start_timestamp": self.metrics.get("test_start_timestamp"),
                    "training_data_sha256": self.metrics.get("training_data_sha256"),
                    "trained_at_utc": self.metrics.get("trained_at_utc"),
                    "environment": self.model_environment,
                    "evaluation_method": self.metrics.get("evaluation_method"),
                    "test_mae_kg_co2_per_hour": self.metrics.get("test_mae"),
                    "test_rmse_kg_co2_per_hour": self.metrics.get("test_rmse"),
                    "test_r2": self.metrics.get("test_r2"),
                },
                "input_quality": {
                    "status": "Required fields complete; source authenticity and freshness are not verified",
                    "completeness_percent": 100.0,
                    "required_input_count": len(REQUIRED_INPUTS),
                    "missing_inputs": [],
                    "range_validation_status": "passed",
                    "source_verification": "not available",
                    "training_range_status": support_status,
                    "outside_training_range": result["model_support"]["affected_features"],
                    "model_support": result["model_support"],
                    "limitations": "Completeness is field presence only. Configured demo bounds are not equipment safety limits; sensor freshness, anomaly, and source authenticity are not checked.",
                },
                "measurement_period": (
                    f"One-hour operating interval at hour {data['hour']:02d}; measurement date was not submitted."
                    if data.get("hour") is not None
                    else "Measurement time was not supplied; this is an estimate from the submitted operating values without temporal context."
                ),
                "production_unit_note": "Production intensity uses the submitted source unit named 'tons'; metric versus short ton is unspecified and no conversion is applied.",
                "feature_contributions": explanation["feature_contributions"] if explanation else [],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "carbon_intensity_kg_co2_per_production_unit": round(result["predicted_emissions_kg_co2"] / production, 3) if result["predicted_emissions_kg_co2"] is not None and production > 0 else None,
        })
        return result

    def explain(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes SHAP waterfall explanation decomposing the prediction into feature contributions.
        Returns φ₀ (baseline average), φᵢ (feature SHAP values), and percentage impacts.
        """
        X = self.prepare_df(data)
        raw_prediction = float(self.model.predict(X)[0])
        model_support = self._assess_model_support(data, X)
        validity = self._prediction_validity(raw_prediction, model_support)
        if not validity["prediction_available"]:
            return {
                "available": False,
                "model_version": self.metrics.get("model_version"),
                "prediction_validity": validity,
                "model_support": model_support,
                "method": "Explanation unavailable for unsupported operating conditions",
                "feature_contributions": [],
                "causality_note": "Feature contributions explain model behavior and do not establish physical causation.",
            }

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
            "available": True,
            "model_version": self.metrics.get("model_version"),
            "prediction_validity": validity,
            "model_support": model_support,
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

        base_support = self._assess_model_support(baseline_data, X_base)
        scenario_support = self._assess_model_support(scenario_data, X_sim)
        base_validity = self._prediction_validity(base_pred, base_support)
        scenario_validity = self._prediction_validity(sim_pred, scenario_support)
        if not base_validity["prediction_available"]:
            raise ValueError(f"Scenario rejected: baseline is unsupported. {base_validity['reason']}")
        if not scenario_validity["prediction_available"]:
            raise ValueError(f"Scenario rejected: proposed operating conditions are unsupported. {scenario_validity['reason']}")
        baseline_interval = self._empirical_prediction_interval(base_pred)
        scenario_interval = self._empirical_prediction_interval(sim_pred)
        
        delta_kg = round(base_pred - sim_pred, 2)
        pct_change = round((delta_kg / (base_pred + 1e-9)) * 100.0, 2)
        baseline_production = float(baseline_data["production_volume_tons"])
        scenario_production = float(scenario_data["production_volume_tons"])
        production_change = scenario_production - baseline_production
        
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
        baseline_reference = calculate_activity_baseline(baseline_data)
        scenario_reference = calculate_activity_baseline(scenario_data)
                
        return {
            "method": "Model-estimated scenario",
            "unit": "kg CO2/hour",
            "model_version": self.metrics.get("model_version"),
            "baseline_prediction_interval": baseline_interval,
            "scenario_prediction_interval": scenario_interval,
            "prediction_validity": {
                "status": "DEGRADED" if "DEGRADED" in (base_validity["status"], scenario_validity["status"]) else "VALID",
                "baseline": base_validity,
                "scenario": scenario_validity,
            },
            "model_support": {"baseline": base_support, "scenario": scenario_support},
            "limitation": "Estimated change only; actual reduction must be verified using post-intervention measurements.",
            "baseline_activity_reference": baseline_reference,
            "scenario_activity_reference": scenario_reference,
            "baseline_activity_based_demo": baseline_reference,
            "scenario_activity_based_demo": scenario_reference,
            "baseline_emission_kg_co2": round(base_pred, 2),
            "simulated_emission_kg_co2": round(sim_pred, 2),
            "baseline_intensity_kg_co2_per_production_unit": round(base_pred / baseline_production, 3) if baseline_production > 0 else None,
            "scenario_intensity_kg_co2_per_production_unit": round(sim_pred / scenario_production, 3) if scenario_production > 0 else None,
            "production_change": {
                "baseline_submitted_units_per_hour": baseline_production,
                "scenario_submitted_units_per_hour": scenario_production,
                "absolute_change_submitted_units_per_hour": round(production_change, 3),
                "percent_change": round(production_change / baseline_production * 100.0, 2) if baseline_production > 0 else None,
            },
            "cost_impact": {
                "status": "UNAVAILABLE",
                "reason": "No site energy tariff or cost data is configured.",
            },
            "scenario_constraints": {
                "status": "CONFIGURED_INPUT_BOUNDS_PASSED",
                "note": "Only broad demo input bounds were checked; site process and equipment safety constraints are not configured.",
            },
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
        validate_operational_inputs(data)
        explanation = self.explain(data)
        if not explanation.get("available", True):
            return []
        contributions = {item["feature"]: item for item in explanation["feature_contributions"]}

        recs = []
        
        # Rule 1: High Grid Emission Factor
        grid_factor = data["grid_emission_factor"]
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
        temp = data["furnace_temp_c"]
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
        energy = data["energy_kwh"]
        prod = data["production_volume_tons"]
        specific_energy = energy / prod if prod > 0 else None
        
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
