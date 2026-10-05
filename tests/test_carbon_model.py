import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.accounting import calculate_activity_baseline, load_factor_registry
from src.backend import app
from src.model import EcoSenseModel
from src.validation import validate_operational_inputs


SAMPLE = {
    "energy_kwh": 2200.0,
    "grid_emission_factor": 0.65,
    "production_volume_tons": 45.0,
    "furnace_temp_c": 1020.0,
    "boiler_pressure_bar": 22.0,
    "shift": 1,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "energy_lag1": 2150.0,
    "production_lag1": 44.0,
    "rolling_avg_energy_3h": 2180.0,
}


class AccountingAndValidationTests(unittest.TestCase):
    def test_activity_baseline_matches_demo_formula(self):
        baseline = calculate_activity_baseline(SAMPLE)
        expected = 2200 * 0.65 + (1020 - 800) * 0.45 + (22 - 10) * 8.5 + 45 * 3.8
        self.assertEqual(baseline["value"], round(expected, 2))
        self.assertAlmostEqual(sum(row["emissions_kg_co2"] for row in baseline["components"]), baseline["value"])
        self.assertEqual(baseline["unit"], "kg CO2/hour")

    def test_factor_registry_is_explicitly_unverified(self):
        registry = load_factor_registry()
        self.assertIn("Unverified", registry["status"])
        self.assertEqual(registry["scope"], "Unclassified")
        self.assertIsNone(registry["factors"]["grid_electricity"]["value"])

    def test_invalid_inputs_and_context_are_rejected(self):
        for changes in ({"energy_kwh": -1}, {"hour": 19}, {"is_weekend": 1}):
            invalid = {**SAMPLE, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_operational_inputs(invalid)


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EcoSenseModel(models_dir="models")

    def test_prediction_trace_and_interval_match_model(self):
        prediction = self.model.predict(SAMPLE)
        explanation = self.model.explain(SAMPLE)
        self.assertEqual(prediction["method"], "ML prediction")
        self.assertEqual(prediction["traceability"]["input_values"]["energy_kwh"], SAMPLE["energy_kwh"])
        self.assertEqual(prediction["traceability"]["activity_based_demo_baseline"]["value"], 1802.0)
        self.assertEqual(prediction["traceability"]["input_quality"]["required_input_count"], 11)
        self.assertLess(abs(prediction["predicted_emissions_kg_co2"] - explanation["predicted_emission_kg_co2"]), 0.02)
        self.assertIn("calibrated", prediction["prediction_interval"]["method"])
        self.assertIsNotNone(prediction["prediction_interval"]["observed_holdout_coverage"])
        self.assertEqual(
            prediction["traceability"]["input_quality"]["training_range_status"],
            "within observed model-fit ranges",
        )
        extrapolated = self.model.predict({**SAMPLE, "energy_kwh": 6000.0})
        self.assertIn("energy_kwh", extrapolated["traceability"]["input_quality"]["outside_training_range"])

    def test_scenario_reports_intensity_and_rejects_invalid_values(self):
        scenario = self.model.simulate(SAMPLE, {"energy_kwh": 2000.0, "production_volume_tons": 40.0})
        self.assertEqual(scenario["method"], "Model-estimated scenario")
        self.assertEqual(scenario["scenario_activity_based_demo"]["value"], 1653.0)
        self.assertEqual(
            scenario["scenario_intensity_kg_co2_per_production_unit"],
            round(scenario["simulated_emission_kg_co2"] / 40.0, 3),
        )
        self.assertAlmostEqual(
            sum(item["model_contribution_change_kg_co2_per_hour"] for item in scenario["parameter_impacts"]),
            scenario["simulated_emission_kg_co2"] - scenario["baseline_emission_kg_co2"],
            delta=0.03,
        )
        with self.assertRaises(ValueError):
            self.model.simulate(SAMPLE, {"energy_kwh": -1})
        with self.assertRaises(ValueError):
            self.model.simulate(SAMPLE, {"unknown_input": 1})

    def test_recommendations_do_not_invent_savings_or_confidence(self):
        recommendations = self.model.generate_recommendations(SAMPLE)
        self.assertTrue(recommendations)
        self.assertTrue(all(item["potential_saving_kg_co2_day"] is None for item in recommendations))
        self.assertTrue(all(item["confidence"] is None for item in recommendations))


class APIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_prediction_requires_inputs_and_returns_trace(self):
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 422)
        response = self.client.post("/api/predict", json=SAMPLE)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["traceability"]["input_values"]["grid_emission_factor"], SAMPLE["grid_emission_factor"])
        self.assertEqual(payload["unit"], "kg CO2/hour")

    def test_report_does_not_claim_accounting_or_compliance(self):
        report = self.client.get("/api/report").json()
        self.assertEqual(report["regulatory_compliance"]["status"], "NOT ASSESSED")
        self.assertIsNone(report["summary"]["accounted_facility_emissions"])
        self.assertIn("Synthetic", report["data_classification"])


class StreamlitInputValidationTests(unittest.TestCase):
    def test_shift_hour_mismatch_is_reported_without_prediction_crash(self):
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        app_test = AppTest.from_file(str(app_path), default_timeout=60).run()
        values = {
            "Energy Consumption": 2200.0,
            "Grid Carbon Intensity": 0.65,
            "Production Output": 45.0,
            "Furnace Temperature": 1020.0,
            "Boiler Pressure": 22.0,
            "Hour of Day": 16,
            "Previous Hour Energy": 2150.0,
            "Previous Hour Production": 44.0,
            "3-Hour Rolling Average Energy": 2180.0,
        }
        for widget in app_test.number_input:
            for label_prefix, value in values.items():
                if widget.label.startswith(label_prefix):
                    widget.set_value(value)
        app_test.selectbox[0].select_index(1)
        app_test.selectbox[1].select_index(3)
        app_test.button[0].click().run()

        self.assertFalse(app_test.exception)
        self.assertTrue(any("Select Shift 2" in error.value for error in app_test.error))
        self.assertFalse(any(metric.label == "ML-predicted emissions" for metric in app_test.metric))


if __name__ == "__main__":
    unittest.main()