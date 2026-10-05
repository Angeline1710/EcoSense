"""Input and scenario validation for the single-plant demonstration workflow."""

import math
from typing import Any, Dict


INPUT_RANGES = {
    "energy_kwh": (0.0, 10000.0, "kWh"),
    "grid_emission_factor": (0.0, 2.0, "kg CO2/kWh"),
    "production_volume_tons": (0.0, 500.0, "tonnes/hour"),
    "furnace_temp_c": (0.0, 2000.0, "degrees C"),
    "boiler_pressure_bar": (0.0, 100.0, "bar"),
    "energy_lag1": (0.0, 10000.0, "kWh"),
    "production_lag1": (0.0, 500.0, "tonnes"),
    "rolling_avg_energy_3h": (0.0, 10000.0, "kWh"),
}

REQUIRED_INPUTS = tuple(INPUT_RANGES) + ("shift", "hour", "day_of_week")
SCENARIO_INPUTS = frozenset({
    "energy_kwh",
    "grid_emission_factor",
    "production_volume_tons",
    "furnace_temp_c",
    "boiler_pressure_bar",
})


def validate_operational_inputs(data: Dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_INPUTS if data.get(name) is None]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}.")

    for name, (minimum, maximum, unit) in INPUT_RANGES.items():
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise ValueError(f"Input rejected: {name} must be numeric in {unit}.") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"Input rejected: {name} must be between {minimum:g} and {maximum:g} {unit}.")

    for name, minimum, maximum in (("shift", 1, 3), ("hour", 0, 23), ("day_of_week", 0, 6)):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"Input rejected: {name} must be an integer between {minimum} and {maximum}.")

    hour = data["hour"]
    expected_shift = 1 if 8 <= hour < 16 else 2 if hour >= 16 else 3
    if data["shift"] != expected_shift:
        raise ValueError(f"Input rejected: shift {data['shift']} does not match hour {hour}; expected shift {expected_shift}.")

    expected_weekend = int(data["day_of_week"] >= 5)
    if "is_weekend" in data and data["is_weekend"] is not None and int(data["is_weekend"]) != expected_weekend:
        raise ValueError("Input rejected: weekend status must match the selected day of week.")


def validate_scenario(baseline: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(modifications) - SCENARIO_INPUTS)
    if unknown:
        raise ValueError(f"Scenario rejected: unsupported parameter(s): {', '.join(unknown)}.")
    scenario = baseline.copy()
    scenario.update(modifications)
    validate_operational_inputs(scenario)
    return scenario