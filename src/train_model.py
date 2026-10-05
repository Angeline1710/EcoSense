import os
import json
import hashlib
import platform
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.data_generator import generate_industrial_dataset
from src.validation import REQUIRED_INPUTS

INPUT_FEATURE_COLS = list(REQUIRED_INPUTS)
RANGE_CHECK_INPUTS = list(REQUIRED_INPUTS)

FEATURE_COLS = [
    "electricity_emissions_proxy",
    "furnace_temp_c",
    "boiler_pressure_bar",
    "production_volume_tons"
]

TARGET_COL = "emissions_kg_co2"

FEATURE_NAMES_FRIENDLY = {
    "electricity_emissions_proxy": "Electricity Emissions (kg CO₂/hr)",
    "furnace_temp_c": "Furnace Temperature (°C)",
    "boiler_pressure_bar": "Boiler Pressure (Bar)",
    "production_volume_tons": "Production Output (Tons/hr)"
}

def build_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build physically meaningful predictors from the submitted measurements."""
    return pd.DataFrame({
        "electricity_emissions_proxy": data["energy_kwh"] * data["grid_emission_factor"],
        "furnace_temp_c": data["furnace_temp_c"],
        "boiler_pressure_bar": data["boiler_pressure_bar"],
        "production_volume_tons": data["production_volume_tons"]
    }, index=data.index)[FEATURE_COLS]

def train_and_evaluate(data_path: str = "data/industrial_emissions.csv", models_dir: str = "models"):
    """
    Trains a physics-informed regression model and evaluates it with chronological validation.
    """
    if not os.path.exists(data_path):
        print(f"[Train] Data file '{data_path}' not found. Generating now...")
        generate_industrial_dataset(output_path=data_path)
        
    df = pd.read_csv(data_path)
    X = build_model_features(df)
    y = df[TARGET_COL]

    test_start = int(len(df) * 0.8)
    calibration_start = int(len(df) * 0.7)
    X_train, y_train = X.iloc[:calibration_start], y.iloc[:calibration_start]
    X_calibration = X.iloc[calibration_start:test_start]
    y_calibration = y.iloc[calibration_start:test_start]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    cv_rmse = []
    for train_indices, validation_indices in TimeSeriesSplit(n_splits=4).split(X_train):
        fold_model = LinearRegression()
        fold_model.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
        fold_predictions = fold_model.predict(X_train.iloc[validation_indices])
        cv_rmse.append(float(mean_squared_error(y_train.iloc[validation_indices], fold_predictions) ** 0.5))

    evaluation_model = LinearRegression().fit(X_train, y_train)
    train_predictions = evaluation_model.predict(X_train)
    test_predictions = evaluation_model.predict(X_test)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    train_rmse = mean_squared_error(y_train, train_predictions) ** 0.5
    test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
    test_mae = mean_absolute_error(y_test, test_predictions)

    calibration_model = LinearRegression().fit(X_train, y_train)
    calibration_predictions = calibration_model.predict(X_calibration)
    calibration_errors = np.abs(y_calibration.to_numpy() - calibration_predictions)
    interval_rank = min(len(calibration_errors), int(np.ceil((len(calibration_errors) + 1) * 0.95)))
    interval_error_q95 = float(np.sort(calibration_errors)[interval_rank - 1])
    test_lower = np.maximum(0.0, test_predictions - interval_error_q95)
    test_upper = test_predictions + interval_error_q95
    test_interval_coverage = float(np.mean((y_test.to_numpy() >= test_lower) & (y_test.to_numpy() <= test_upper)))

    # Keep the deployed estimator identical to the one used to create calibration residuals.
    model = evaluation_model
    
    print("=" * 60)
    print("ECOSENSE MODEL TRAINING & CHRONOLOGICAL EVALUATION")
    print("=" * 60)
    print(f"Expanding-window CV RMSE: {np.mean(cv_rmse):.2f} ± {np.std(cv_rmse):.2f} kg CO2")
    print(f"Train R^2 Score: {train_r2:.4f} | Test R^2 Score: {test_r2:.4f}")
    print(f"Train RMSE:     {train_rmse:.2f} kg CO2 | Test RMSE:     {test_rmse:.2f} kg CO2")
    print(f"Test MAE:       {test_mae:.2f} kg CO2")
    print("=" * 60)

    background = pd.DataFrame([X_train.mean()], columns=FEATURE_COLS)
    base_value = float(model.intercept_ + np.dot(model.coef_, background.iloc[0].to_numpy()))
    print(f"SHAP Baseline Value (training-feature mean): {base_value:.2f} kg CO2")
    absolute_training_contributions = np.abs(
        (X_train[FEATURE_COLS].to_numpy() - background.iloc[0].to_numpy()) * np.asarray(model.coef_)
    )
    mean_absolute_contributions = absolute_training_contributions.mean(axis=0)
    total_mean_absolute_contribution = float(mean_absolute_contributions.sum())
    global_feature_importance = [
        {
            "feature": col,
            "friendly_name": FEATURE_NAMES_FRIENDLY[col],
            "mean_absolute_contribution_kg_co2_per_hour": float(value),
            "relative_share_percent": float(value / total_mean_absolute_contribution * 100.0) if total_mean_absolute_contribution else 0.0,
            "method": "Mean absolute additive linear-model contribution over the fit window; not causal importance.",
        }
        for col, value in zip(FEATURE_COLS, mean_absolute_contributions)
    ]
    global_feature_importance.sort(key=lambda item: item["mean_absolute_contribution_kg_co2_per_hour"], reverse=True)

    environment_manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "model_artifact": "ecosense_xgb.joblib",
        "model_version": "EcoSense-PIR-1.0.0",
        "environment_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    elasticity_dict = {}
    for col, coef in zip(FEATURE_COLS, model.coef_):
        elasticity_dict[col] = {
            "coefficient": float(coef),
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "friendly_name": FEATURE_NAMES_FRIENDLY[col]
        }

    os.makedirs(models_dir, exist_ok=True)

    model_filepath = os.path.join(models_dir, "ecosense_xgb.joblib")
    joblib.dump(model, model_filepath)

    with open(data_path, "rb") as data_file:
        training_data_sha256 = hashlib.sha256(data_file.read()).hexdigest()

    metrics = {
        "model_type": "Physics-informed Linear Regression + exact additive SHAP contributions",
        "training_method": "Linear regression over electricity, thermal, and production features",
        "evaluation_method": "Chronological 70/10/20 fit/calibration/test split; four-fold expanding-window CV on fit rows",
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "prediction_interval_method": "95% empirical residual interval calibrated on the preceding chronological 10% window; lower bound truncated at zero; future coverage is not guaranteed",
        "prediction_interval_nominal_coverage": 0.95,
        "prediction_interval_holdout_coverage": test_interval_coverage,
        "prediction_interval_abs_error_q95": interval_error_q95,
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse)),
        "base_value": base_value,
        "n_samples": len(df),
        "model_training_rows": len(X_train),
        "calibration_rows": len(X_calibration),
        "test_rows": len(X_test),
        "input_feature_count": len(INPUT_FEATURE_COLS),
        "calibration_start_timestamp": str(df["timestamp"].iloc[calibration_start]) if "timestamp" in df else str(calibration_start),
        "test_start_timestamp": str(df["timestamp"].iloc[test_start]) if "timestamp" in df else str(test_start),
        "model_version": "EcoSense-PIR-1.0.0",
        "feature_schema_version": "1.0.0",
        "training_data_sha256": training_data_sha256,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_environment": environment_manifest,
        "global_feature_importance": global_feature_importance,
        "feature_cols": FEATURE_COLS,
        "feature_names_friendly": FEATURE_NAMES_FRIENDLY
    }
    metrics["training_input_ranges"] = {
        col: {"min": float(df.iloc[:calibration_start][col].min()), "max": float(df.iloc[:calibration_start][col].max())}
        for col in RANGE_CHECK_INPUTS
    }
    metrics["model_feature_ranges"] = {
        col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
        for col in FEATURE_COLS
    }
    
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(os.path.join(models_dir, "elasticity.json"), "w") as f:
        json.dump(elasticity_dict, f, indent=2)

    with open(os.path.join(models_dir, "model_environment.json"), "w") as f:
        json.dump(environment_manifest, f, indent=2)

    print(f"[Train] Model artifacts successfully saved to '{models_dir}/'.")
    return model, metrics

if __name__ == "__main__":
    train_and_evaluate()
