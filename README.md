# EcoSense

EcoSense is an industrial emissions intelligence MVP that helps operations and sustainability teams estimate carbon output, understand the main drivers behind a prediction, and test what-if changes before making process decisions.

The project combines a physics-informed linear regression model, SHAP explanations, a synthetic activity-formula baseline, a model-based scenario simulator, and two lightweight interfaces. It remains a synthetic-data prototype, not a live carbon-accounting or IoT system.

## What the system does

- Predicts hourly emissions from operational inputs such as energy use, grid intensity, production volume, furnace temperature, and shift conditions.
- Separates the synthetic activity-based demo baseline from the ML prediction and scenario estimate.
- Explains model feature contributions with SHAP; these explain model behavior, not physical causality.
- Simulates explicitly submitted scenarios, validates configured bounds, and reports intensity alongside absolute estimates.
- Generates heuristic review prompts without inventing savings, cost, production impact, or confidence values.
- Provides both a landing page and a dashboard experience.

## App entry points

- Homepage: http://127.0.0.1:8000/
- Dashboard: http://127.0.0.1:8000/dashboard
- Streamlit demo: http://localhost:8501
- API docs: http://127.0.0.1:8000/docs

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Generate the demo dataset

```bash
python -m src.data_generator
```

This creates the synthetic industrial emissions dataset in `data/industrial_emissions.csv`.

### 3) Train the model

```bash
python -m src.train_model
```

This trains the regression model and saves artifacts in `models/`. Evaluation uses chronological 70% fit, 10% calibration, and 20% untouched final-test windows, plus four expanding-window validation folds over the fit period. The deployed estimator remains the fit-window model used to create the empirical residual interval.

### 4) Run the app

#### Option A: Streamlit demo

```bash
streamlit run streamlit_app.py
```

#### Option B: FastAPI web app

```bash
python run_server.py
```

Then open the local routes listed above.

## Project structure

```text
EcoSense/
├── data/
│   ├── industrial_emissions.csv
│   └── emission_factor_registry.json
├── frontend/
│   ├── static/
│   │   └── css/
│   │       └── style.css
│   └── templates/
│       ├── home.html
│       └── index.html
├── models/
│   ├── ecosense_xgb.joblib  # Legacy filename; currently stores linear regression
│   ├── elasticity.json
│   ├── metrics.json
│   └── scaler.pkl
├── src/
│   ├── accounting.py
│   ├── __init__.py
│   ├── backend.py
│   ├── data_generator.py
│   ├── model.py
│   ├── train_model.py
│   └── validation.py
├── tests/
│   └── test_carbon_model.py
├── EcoSense_PRD.md
├── README.md
├── requirements.txt
├── run_server.py
├── streamlit_app.py
└── .gitignore
```

## Core components

### Backend API
The FastAPI server in `src/backend.py` exposes:

- `GET /` — overview landing page
- `GET /dashboard` — HTML dashboard
- `GET /api/status` — model health and metrics
- `GET /api/data/recent` — recent synthetic demo records for charts
- `POST /api/predict` — emissions prediction
- `POST /api/explain` — SHAP-based feature explanation
- `POST /api/simulate` — what-if scenario comparison
- `POST /api/recommend` — action-oriented carbon recommendations
- `GET /api/report` — summary ESG-style report payload

### Model engine
`src/accounting.py` calculates the synthetic activity baseline using the documented demo assumptions.

`src/validation.py` checks required inputs, configured demo bounds, shift/time consistency, and allowed scenario fields.

`src/model.py` contains the main EcoSenseModel logic:

- loads trained artifacts from `models/`
- prepares feature data
- predicts emissions from electricity, thermal, and production features using linear regression
- computes SHAP feature contributions
- runs model-based scenario inference and intensity comparisons
- generates recommendation rules

### Frontend
- `frontend/templates/home.html` provides the landing overview page.
- `frontend/templates/index.html` provides the main dashboard UI.
- `streamlit_app.py` provides the interactive Streamlit demo experience.

## Model performance

Latest training results on the bundled synthetic dataset (2,160 hourly records):

- Expanding-window CV RMSE: 20.18 ± 0.76 kg CO₂/hr
- Final 20% chronological holdout R²: 0.99858
- Final 20% chronological holdout RMSE: 20.22 kg CO₂/hr
- Final 20% chronological holdout MAE: 15.99 kg CO₂/hr
- Nominal 95% residual interval coverage on the final test period: 95.14% (432 rows)
- Model type: Physics-informed Linear Regression + SHAP Explainer

The model uses four engineered features: energy consumption multiplied by the submitted grid factor, furnace temperature, boiler pressure, and production volume. Intensity divides by the submitted production value without unit conversion because the source column's “tons” basis is unspecified. The empirical interval uses residuals from the preceding 10% calibration window; the final 20% is excluded from fitting and calibration. These results apply only to generated demo data, not real facility measurements.

## Emissions and limitations

- The activity formula is labelled **Activity-based demo baseline**, not accounted facility emissions. Electricity intensity is user supplied; thermal and production coefficients are synthetic generator assumptions.
- `data/emission_factor_registry.json` records the demo assumptions, unverified provenance, unknown geography/scope, and unavailable effective dates. These are not official reporting factors.
- The API report separates synthetic labels, model metrics, accounted emissions (unavailable), and post-intervention verification (unavailable). It does not claim regulatory compliance.
- Input completeness means required fields were supplied and passed configured demo bounds. It does not verify source authenticity, sensor freshness, anomalies, or equipment safety limits.
- Recommendation rules are screening prompts. Savings, cost, production impact, and confidence are reported as unavailable unless measured/modelled evidence exists.
- `models/ecosense_xgb.joblib` retains its historical filename for compatibility; the current estimator is linear regression. `models/metrics.json` identifies the active model.

## Notes

- This project uses synthetic demo data and is intended for prototyping and presentation.
- It does not yet connect to live factory sensors or enterprise monitoring systems.
- It demonstrates predict → explain → simulate → recommend, but operational actions require expert review and measured post-change verification.
- It does not yet provide authoritative factors, regulatory scope mapping, drift detection, anomaly detection, cost optimization, persistent audit records, or an estimate-to-actual feedback loop.

## Typical workflow

1. Open the overview page at the root URL.
2. Choose the demo or dashboard entry.
3. Submit operational values.
4. Review predicted emissions and key drivers.
5. Adjust a scenario to test improvements.
6. Use the recommendations to guide operational actions.

## License

This project is intended for internal prototype and demo use unless otherwise specified by the repository owner.
