# Product Requirements Document (PRD)
## EcoSense — AI-Powered Carbon Optimization Copilot for Smart Industries
**Team EcoX | v3.0 — 1-Week MVP Scope, with Detailed Model Construction**

---

## 1. Overview

EcoSense is a real-time carbon emission prediction and explainability tool for industrial manufacturing. This MVP proves the core loop — **predict emissions → explain why → simulate a change** — using a realistic dataset, within a 1-week build window, as a working demonstration ahead of full-scale, hardware-integrated deployment.

## 2. Problem Statement

Most factories track emissions manually or through periodic reports, discovering impact only after the fact. As ESG regulations tighten, this delay is a compliance and competitiveness risk. EcoSense demonstrates a system that predicts emissions early, explains the cause, and estimates the impact of corrective action — without needing live factory hardware to prove the concept.

## 3. Goals (MVP)

- Predict near-term carbon emissions from operational data.
- Explain each prediction in terms of contributing factors.
- Let users simulate a proposed change and see estimated impact.
- Present all of this through a clear, functioning dashboard.

## 4. Target Users

- **Factory operations managers** — day-to-day monitoring and action.
- **Sustainability/ESG officers** — compliance and reporting.
- **Plant engineers** — root-cause investigation of emission spikes.

## 5. MVP Scope

| Included (Build This Week) | Excluded (Post-MVP) |
|---|---|
| Emission prediction model | Live IoT/sensor integration |
| SHAP-based explainability | Full LLM Copilot (live GPT/Llama calls) |
| Rule-based what-if simulator | Dynamic ESG report generation |
| Single-page dashboard | Alerting/notification system |
| Templated recommendation text | Multi-factory/multi-tenant support |

*Rationale: the MVP proves the hardest and most differentiating parts — prediction accuracy and explainability — while substituting lightweight stand-ins (templated text, static reports, synthetic data) for components that need more time or real-world access to build properly.*

## 6. 7-Day Build Plan

| Day | Focus | Output |
|---|---|---|
| 1 | Data sourcing & cleaning | Structured dataset (energy, production, timestamp, derived emissions) |
| 2 | Prediction model | Trained regression model + `/predict` API endpoint |
| 3 | Explainability | SHAP integration + `/explain` API endpoint |
| 4 | What-if simulator | Coefficient-based `/simulate` endpoint |
| 5 | Dashboard | React/Streamlit frontend with 3 panels: predictions, explanations, simulator |
| 6 | Integration + recommendations | End-to-end connection + templated recommendation text |
| 7 | Polish & rehearsal | Bug fixes, mock ESG report, demo scenarios, pitch rehearsal |

**Critical path:** Day 1 (data) must not slip — every later step depends on it. If time runs short, reduce the What-If Simulator's precision before cutting SHAP explainability, since explainability is the core credibility factor for the pitch.

## 7. Model Construction — Detailed Approach

### 7.1 Data Preprocessing
Raw operational data is rarely model-ready. This stage includes:
- **Missing value imputation** — forward-fill or interpolation for sensor gaps, common in real industrial datasets.
- **Normalization/scaling** — Min-Max or Z-score scaling on continuous features (energy, volume) so the model isn't biased toward large-magnitude variables.
- **Feature engineering** — lag features (previous hour's energy use), rolling averages, shift/hour/day-of-week encodings, and a derived target emissions column computed as `Energy (kWh) × Grid Emission Factor (kg CO₂/kWh)`.
- **Train/test split** — 80:20, with a held-out validation set for hyperparameter tuning.

### 7.2 Baseline Model Selection — Why XGBoost
For the MVP, **XGBoost (Extreme Gradient Boosting)** is the practical choice over deep learning:
- It sequentially improves predictive performance through ensemble tree learning, handling nonlinear relationships between energy, production, and emissions well without needing large datasets.
- It is the standard baseline in recent industrial carbon prediction research, valued for high accuracy and computational efficiency.
- It trains in minutes, not hours — critical for a 1-week build.

Key hyperparameters: `max_depth`, `n_estimators`, `learning_rate (eta)`, `subsample`, `reg_lambda`. For MVP, light grid search (3–4 values each) is sufficient rather than full Bayesian optimization.

### 7.3 Model Evaluation
Performance is judged using **R²** (variance explained) and **RMSE** (average error in kg CO₂) — RMSE is more interpretable than R² alone for a non-technical government audience.

### 7.4 Explainability — SHAP Integration
SHAP is layered on top of the trained model (not a separate model) to decompose each prediction:

**Ĉ = φ₀ + Σ φᵢ**

where φ₀ is the baseline prediction (average across all samples) and φᵢ is the SHAP value of feature *i* — its contribution to that specific prediction, computed via the Shapley value from cooperative game theory, which fairly distributes "credit" across features based on all possible feature combinations.

Practically, this produces a ranked breakdown per prediction — e.g., "Energy consumption: +18%, Shift timing: +9%, Grid intensity: +6%" — which becomes the explainability panel on the dashboard. This is the bridge between raw prediction and actionable recommendation.

### 7.5 What-If Simulation Logic
Rather than retraining per query, the simulator uses precomputed elasticity coefficients — correlations between feature changes and emission changes, derived from the same training data. This is fast, deterministic, and safe to demo live without model drift.

### 7.6 End-to-End Process Flow

```
Raw sensor/dataset input
        ↓
Data cleaning + feature engineering
        ↓
XGBoost regression model → Emission prediction (with RMSE-based confidence range)
        ↓
SHAP explainer → Per-feature contribution breakdown
        ↓
What-if simulator → Estimated impact of proposed change (elasticity-based)
        ↓
Templated recommendation → Dashboard display
```

### 7.7 Post-MVP Scaling Path
Current research shows that for higher accuracy at scale, teams commonly upgrade from XGBoost to hybrid deep learning approaches — e.g., BiLSTM-based frameworks have shown notable RMSE improvement over ARIMA, Linear Regression, and XGBoost on daily plant-level data, and hybrid LSTM-XGBoost models combine temporal pattern learning with feature-based relationships for hourly industrial data. XGBoost remains the right MVP choice for speed and interpretability; LSTM/BiLSTM is documented here as a planned upgrade path once more historical time-series data is available.

## 8. Tech Stack (MVP-scoped)

**Data & Backend**
- **Python** — data processing and ML pipeline
- **Pandas / NumPy** — cleaning and feature engineering
- **Public/synthetic dataset** (e.g., UCI Steel Industry Energy Consumption) in place of live sensor data
- **FastAPI** — lightweight API layer for predict/explain/simulate endpoints
- **CSV/SQLite or PostgreSQL** — simple data storage (no need for time-series DB at MVP scale)

**Machine Learning**
- **Scikit-learn / XGBoost** — regression model for emission prediction
- **SHAP** — explainability layer
- **Basic correlation coefficients** (NumPy/Pandas) — what-if simulation logic, no retraining required

**Recommendation Layer (MVP stand-in)**
- Templated string logic (if-else / f-strings based on SHAP output) instead of a live LLM call — structured to be swapped for GPT/Llama post-MVP with minimal rework

**Frontend**
- **Streamlit** (fastest path) or **React.js** if time allows — single dashboard with prediction chart, explainability breakdown, and simulator input

**Infrastructure**
- Local or single cloud instance (AWS/GCP free tier) — no need for containerized multi-service deployment at MVP stage

## 9. Success Criteria for MVP Demo

- Model produces a reasonable emission prediction from sample input.
- Explainability panel clearly attributes prediction to specific factors.
- Simulator returns a plausible estimated impact for a hypothetical change.
- Dashboard runs live, end-to-end, without errors, across 2–3 rehearsed demo scenarios.

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| No access to real factory data | Use public/synthetic dataset with a derived emissions column |
| Time overrun on any single day | Cut lowest-priority feature first (simulator precision → LLM stand-in → report polish), never cut explainability |
| Dashboard integration issues late in the week | Build backend endpoints independently and test with Postman/curl before wiring up frontend, to isolate integration bugs early |

## 11. Post-MVP Roadmap (Not in Scope This Week)

- Upgrade prediction model from XGBoost to hybrid LSTM/BiLSTM-XGBoost for higher time-series accuracy
- Real IoT/sensor data integration
- Live LLM Copilot (GPT/Llama) replacing templated recommendations
- Automated, audit-ready ESG report generation
- Alerting and multi-factory support

## 12. Conclusion

This MVP is scoped to demonstrate EcoSense's core value — real-time, explainable emission prediction with actionable simulation — within a realistic 1-week build. The model construction approach (XGBoost + SHAP, with a documented path to hybrid deep learning) is grounded in current industrial emission forecasting research, giving the pitch technical credibility while remaining feasible to implement, test, and demo within the available time.
