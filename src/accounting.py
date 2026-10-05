"""Activity-based demo baseline calculations and factor provenance."""

import json
import os
from typing import Any, Dict

import numpy as np
from src.validation import validate_operational_inputs


REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "emission_factor_registry.json",
)


def load_factor_registry() -> Dict[str, Any]:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as registry_file:
        return json.load(registry_file)


def _factor_provenance(factor: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "factor_id": factor["factor_id"],
        "factor_source": factor["source"],
        "factor_geography": factor["geography"],
        "factor_scope": factor["scope"],
        "factor_version": factor["version"],
        "factor_effective_from": factor["effective_from"],
        "factor_effective_to": factor["effective_to"],
        "factor_verification_status": factor["verification_status"],
        "factor_uncertainty": factor["uncertainty"],
        "factor_methodology": factor["methodology"],
        "registry_factor_value": factor["value"],
    }


def calculate_demo_activity_components(
    energy_kwh: Any,
    grid_emission_factor: Any,
    furnace_temp_c: Any,
    boiler_pressure_bar: Any,
    production_volume: Any,
) -> list[Dict[str, Any]]:
    """Calculate shared demo components; thermal proxies represent excess above reference only."""
    factors = load_factor_registry()["factors"]
    furnace_factor = factors["furnace_temperature_proxy"]
    boiler_factor = factors["boiler_pressure_proxy"]
    production_factor = factors["production_process_proxy"]
    grid_factor = factors["grid_electricity"]

    furnace_excess = np.maximum(
        np.asarray(furnace_temp_c, dtype=float) - furnace_factor["base_value"], 0.0
    )
    boiler_excess = np.maximum(
        np.asarray(boiler_pressure_bar, dtype=float) - boiler_factor["base_value"], 0.0
    )

    return [
        {
            "source": grid_factor["category"],
            "activity": energy_kwh,
            "activity_unit": grid_factor["activity_unit"],
            "emission_factor": grid_emission_factor,
            "factor_unit": grid_factor["unit"],
            **_factor_provenance(grid_factor),
            "emissions_kg_co2": np.asarray(energy_kwh, dtype=float) * np.asarray(grid_emission_factor, dtype=float),
        },
        {
            "source": furnace_factor["category"],
            "activity": furnace_excess,
            "activity_unit": furnace_factor["activity_unit"],
            "emission_factor": furnace_factor["value"],
            "factor_unit": furnace_factor["unit"],
            **_factor_provenance(furnace_factor),
            "emissions_kg_co2": furnace_excess * furnace_factor["value"],
        },
        {
            "source": boiler_factor["category"],
            "activity": boiler_excess,
            "activity_unit": boiler_factor["activity_unit"],
            "emission_factor": boiler_factor["value"],
            "factor_unit": boiler_factor["unit"],
            **_factor_provenance(boiler_factor),
            "emissions_kg_co2": boiler_excess * boiler_factor["value"],
        },
        {
            "source": production_factor["category"],
            "activity": production_volume,
            "activity_unit": production_factor["activity_unit"],
            "emission_factor": production_factor["value"],
            "factor_unit": production_factor["unit"],
            **_factor_provenance(production_factor),
            "emissions_kg_co2": np.asarray(production_volume, dtype=float) * production_factor["value"],
        },
    ]


class CarbonAccountingEngine:
    """Single source for explicitly synthetic activity-reference calculations."""

    def calculate_activity_reference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        validate_operational_inputs(data)
        registry = load_factor_registry()
        production = float(data["production_volume_tons"])
        components = calculate_demo_activity_components(
            data["energy_kwh"],
            data["grid_emission_factor"],
            data["furnace_temp_c"],
            data["boiler_pressure_bar"],
            data["production_volume_tons"],
        )
        for component in components:
            for key in ("activity", "emission_factor", "emissions_kg_co2"):
                component[key] = float(np.asarray(component[key]).item())

        total = sum(component["emissions_kg_co2"] for component in components)
        for component in components:
            component["share_percent"] = round(component["emissions_kg_co2"] / total * 100.0, 2) if total else 0.0

        return {
            "label": "Synthetic Activity Reference",
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


_ACCOUNTING_ENGINE = CarbonAccountingEngine()


def calculate_activity_baseline(data: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible wrapper for the synthetic activity reference."""
    return _ACCOUNTING_ENGINE.calculate_activity_reference(data)