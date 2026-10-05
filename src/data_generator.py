import os
import numpy as np
import pandas as pd
from src.accounting import calculate_demo_activity_components

def generate_industrial_dataset(output_path: str = "data/industrial_emissions.csv", days: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Generates controlled synthetic hourly demonstration data, not measured facility observations.
    """
    np.random.seed(seed)
    n_hours = days * 24
    
    # 1. Timestamps
    start_date = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [start_date + pd.Timedelta(hours=i) for i in range(n_hours)]
    df = pd.DataFrame({"timestamp": timestamps})
    
    # Time attributes
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Shift assignment
    # Shift 1: 08:00 - 16:00 (Peak daytime production)
    # Shift 2: 16:00 - 00:00 (Evening production)
    # Shift 3: 00:00 - 08:00 (Night production / lower volume)
    def assign_shift(h):
        if 8 <= h < 16:
            return 1
        elif 16 <= h < 24:
            return 2
        else:
            return 3
            
    df["shift"] = df["hour"].apply(assign_shift)
    
    # 2. Operational parameters with realistic diurnal patterns
    # Grid emission factor: Lower during mid-day (solar grid mix, ~0.42 kg CO2/kWh), higher at night (~0.72 kg CO2/kWh)
    base_grid_ef = 0.58 + 0.14 * np.sin((df["hour"] - 4) * np.pi / 12) + np.random.normal(0, 0.02, n_hours)
    df["grid_emission_factor"] = np.clip(base_grid_ef, 0.35, 0.85)
    
    # Production volume (tons per hour)
    # Higher during Shift 1 & 2, lower on weekends
    base_prod = 38.0 + 12.0 * (df["shift"] == 1) + 6.0 * (df["shift"] == 2) - 10.0 * df["is_weekend"]
    df["production_volume_tons"] = np.clip(base_prod + np.random.normal(0, 3.5, n_hours), 10.0, 75.0)
    
    # Energy consumption (kWh) - Strongly correlated with production + furnace heat baseline
    base_energy = 1000.0 + (df["production_volume_tons"] * 45.0) + (df["shift"] == 1) * 300.0
    df["energy_kwh"] = np.clip(base_energy + np.random.normal(0, 120.0, n_hours), 800.0, 4500.0)
    
    # Furnace temperature (°C)
    base_temp = 950.0 + (df["production_volume_tons"] * 4.0) + np.random.normal(0, 25.0, n_hours)
    df["furnace_temp_c"] = np.clip(base_temp, 750.0, 1300.0)
    
    # Boiler pressure (bar)
    base_press = 18.0 + (df["energy_kwh"] / 300.0) + np.random.normal(0, 1.2, n_hours)
    df["boiler_pressure_bar"] = np.clip(base_press, 10.0, 32.0)
    
    # 3. Lag features and rolling averages (feature engineering)
    df["energy_lag1"] = df["energy_kwh"].shift(1).bfill()
    df["production_lag1"] = df["production_volume_tons"].shift(1).bfill()
    df["rolling_avg_energy_3h"] = df["energy_kwh"].rolling(window=3, min_periods=1).mean()
    
    # 4. Target labels reuse the same synthetic activity formula as runtime accounting.
    components = calculate_demo_activity_components(
        df["energy_kwh"],
        df["grid_emission_factor"],
        df["furnace_temp_c"],
        df["boiler_pressure_bar"],
        df["production_volume_tons"],
    )
    activity_emissions = sum(component["emissions_kg_co2"] for component in components)
    noise = np.random.normal(0, 20.0, n_hours)
    df["emissions_kg_co2"] = np.round(np.maximum(activity_emissions + noise, 0.0), 2)
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Data Generator] Successfully generated {len(df)} records saved to '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_industrial_dataset()
