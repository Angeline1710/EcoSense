"""Activity-based demo baseline calculations and factor provenance."""

import json
import os
from typing import Any, Dict


REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "emission_factor_registry.json",
)


def load_factor_registry() -> Dict[str, Any]:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as registry_file:
        return json.load(registry_file)


def calculate_activity_baseline(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the synthetic activity baseline using only recorded demo assumptions."""
    registry = load_factor_registry()
    factors = registry["factors"]

    electricity_activity = float(data["energy_kwh"])
    grid_factor = float(data["grid_emission_factor"])
    furnace_temperature = float(data["furnace_temp_c"])
    boiler_pressure = float(data["boiler_pressure_bar"])
    production = float(data["production_volume_tons"])

    components = [
        {
            "source": "Grid electricity",
            "activity": electricity_activity,
            "activity_unit": "kWh",
            "emission_factor": grid_factor,
            "factor_unit": "kg CO2/kWh",
            "factor_id": factors["grid_electricity"]["factor_id"],
            "factor_source": factors["grid_electricity"]["source"],
            "emissions_kg_co2": electricity_activity * grid_factor,
        },
        {
            "source": "Thermal process proxy",
            "activity": furnace_temperature - factors["furnace_temperature_proxy"]["base_value"],
            "activity_unit": "degree C above 800 C",
            "emission_factor": factors["furnace_temperature_proxy"]["value"],
            "factor_unit": factors["furnace_temperature_proxy"]["factor_unit"],
            "factor_id": factors["furnace_temperature_proxy"]["factor_id"],
            "factor_source": factors["furnace_temperature_proxy"]["source"],
            "emissions_kg_co2": (furnace_temperature - 800.0) * factors["furnace_temperature_proxy"]["value"],
        },
        {
            "source": "Boiler pressure proxy",
            "activity": boiler_pressure - factors["boiler_pressure_proxy"]["base_value"],
            "activity_unit": "bar above 10 bar",
            "emission_factor": factors["boiler_pressure_proxy"]["value"],
            "factor_unit": factors["boiler_pressure_proxy"]["factor_unit"],
            "factor_id": factors["boiler_pressure_proxy"]["factor_id"],
            "factor_source": factors["boiler_pressure_proxy"]["source"],
            "emissions_kg_co2": (boiler_pressure - 10.0) * factors["boiler_pressure_proxy"]["value"],
        },
        {
            "source": "Production process proxy",
            "activity": production,
            "activity_unit": "dataset production units per hour (source field named tons)",
            "emission_factor": factors["production_process_proxy"]["value"],
            "factor_unit": factors["production_process_proxy"]["factor_unit"],
            "factor_id": factors["production_process_proxy"]["factor_id"],
            "factor_source": factors["production_process_proxy"]["source"],
            "emissions_kg_co2": production * factors["production_process_proxy"]["value"],
        },
    ]

    total = sum(component["emissions_kg_co2"] for component in components)
    for component in components:
        component["share_percent"] = round(component["emissions_kg_co2"] / total * 100.0, 2) if total else 0.0

    return {
        "label": "Activity-based demo baseline",
        "method": "Synthetic activity formula; not verified carbon accounting",
        "value": round(total, 2),
        "unit": "kg CO2/hour",
        "production_intensity_kg_co2_per_production_unit": round(total / production, 3) if production > 0 else None,
        "factor_registry": registry["registry_name"],
        "factor_registry_version": registry["registry_version"],
        "factor_registry_status": registry["status"],
        "geography": registry["geography"],
        "scope": registry["scope"],
        "production_unit_note": registry["production_unit_note"],
        "components": components,
        "limitations": "Synthetic demo coefficients; no authoritative factors, source scopes, or facility verification are configured.",
    }