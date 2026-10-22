# CityPulse: TrafficPulse AI — Predict. Explain. Act.
### Enterprise Highway Traffic Intelligence, Multi-Horizon Forecasting & Decision Support

CityPulse is an end-to-end, reproducible traffic intelligence and decision-support platform built with **Streamlit**, **Scikit-Learn**, **PyArrow**, **pandas**, and **Plotly**.

---

## 🌟 Architecture & Geographic Scope

- **Primary Benchmark Dataset:** `data/train.parquet` (157 MB, 4,962,618 observations across 50 columns).
- **Monitored Domain:** **Los Angeles County Highway Network, California, USA** (Caltrans Performance Measurement System - PeMS District 7).
- **Sensor Network:** 207 inductive loop detector stations with verified geographic coordinates (Lat 34.04° to 34.22° N, Lon -118.53° to -118.18° W).
- **Corridors Monitored:** US-101 (Hollywood Fwy), I-405 (San Diego Fwy), SR-134 (Ventura Fwy), I-110 (Harbor Fwy / Downtown Interchange), I-5 (Golden State Fwy), I-210 (Foothill Fwy), and I-10 (Santa Monica Fwy).
- **Temporal Resolution:** Uniform 5-minute sampling intervals spanning March 1, 2012 to May 23, 2012 (~83.2 consecutive days).
- **Data Integrity:** Historical data is explicitly labeled with exact date ranges (March 1 – May 23, 2012) and never misrepresented as live streaming traffic.

---

## 🚀 The Three Integrated Core Features

### 1. Dynamic Traffic Hotspot Ranking (`src/hotspot_ranking.py`)
- For every sensor or verified road segment, computes:
  - Current observed speed ($t=0$) and predicted speed 15 minutes ahead ($t+15\text{m}$).
  - Speed deficit relative to free-flow reference (65 mph).
  - Recent speed deterioration over 15 and 30-minute horizons.
  - Historical congestion frequency and persistence index.
  - Normalized priority score from 0 to 100 with configurable component weights:
    $$\text{Priority Score} = 100 \times \left(0.30 \times S_{\text{sev}} + 0.25 \times S_{\text{det}} + 0.25 \times S_{\text{rec}} + 0.20 \times S_{\text{per}}\right)$$
- Assigns configurable severity tiers:
  - **Critical:** $\ge 75$
  - **High:** $\ge 50$
  - **Medium:** $\ge 25$
  - **Low:** $< 25$
- Deterministic, explainable ranking with Top 10 hotspot extraction and precision@10 validation.

### 2. Recurring Hotspot & Rush-Hour Intelligence (`src/traffic_patterns.py`)
- Historical traffic recurrence analysis across the full Caltrans PeMS dataset:
  - $24 \times 7$ Hour-of-Day $\times$ Day-of-Week congestion heatmap.
  - Diurnal average speed curves and peak congestion windows (Morning: 07:00–10:00, Midday: 11:00–14:00, Evening: 16:00–19:00).
  - Persistent hotspot identification ($\ge 15\%$ historical observation frequency below 35 mph).
  - **Automated Evidence-Backed Insights:** Generated dynamically using the strict validation template:
    `"Sensor {sensor_id} recorded an average speed of {observed_speed} mph during {period}, compared with {baseline_speed} mph during {comparison_period}."`

### 3. Explainable Intervention Planner (`src/intervention_planner.py` & `src/scenario_engine.py`)
- Synthesizes predictions, historical patterns, and kinematic shockwaves to answer **six core operational questions**:
  1. *What is happening?* (Current speeds, conditions, and deficits).
  2. *What is predicted to happen in 15 minutes?* (Forecasted speeds and trajectory).
  3. *Why is the hotspot highly ranked?* (Component score decomposition).
  4. *Which metrics support the explanation?* (Speed deficit, trend, recurrence).
  5. *Which action should be investigated?* (Operational playbook triggers: Ramp metering, Freeway Service Patrol, Dynamic Message Sign advisory).
  6. *What uncertainty or limitation applies?* (Model confidence and historical variance disclaimers).
- **What-If Simulation Controls:** Capacity throttling (0–100%), duration (15–120 min), and adverse weather shocks with graph shockwave spillover (`data/adj_mx.pkl`).
- **Interactive Weight Sensitivity Explorer:** Dynamically adjust component weights ($w_{\text{sev}}, w_{\text{det}}, w_{\text{rec}}, w_{\text{per}}$) and severity thresholds to observe ranking reordering without model retraining.

---

## 📊 Held-Out Test & Backtest Evaluation Results

All final evaluation metrics are computed on held-out test data from May 1 to May 23, 2012 ($>= \text{2012-05-01}$), maintaining zero target leakage.

### Classification Benchmark (Congestion Level at $t+15\text{m}$)
| Model Architecture | Test Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Severe Congestion Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 77.13% | 33.33% | 0.2903 | 0.6719 | 0.00% | Reference |
| **Logistic Regression (L2)** | 87.01% | 61.35% | 0.6615 | 0.8524 | 48.20% | Linear Baseline |
| **HistGradientBoostingClassifier** | **90.00%** | **77.74%** | **0.7926** | **0.8953** | **81.50%** | **Selected Champion** |
| **Alternative: LightGBM Classifier** | 89.92% | 77.20% | 0.7891 | 0.8942 | 81.10% | Candidate |

### Speed Forecasting Benchmark (Target: $y_{t+3, d0}$ Continuous Speed at $t+15\text{m}$)
| Model Architecture | Test MAE (mph) | Test RMSE (mph) | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline ($y_{t+3} = x_{t+0}$)** | 3.74 mph | 8.08 mph | 0.7405 | Domain Baseline |
| **Ridge Regression (L2)** | 3.80 mph | 7.49 mph | 0.7765 | Linear Regularized |
| **HistGradientBoostingRegressor** | **3.66 mph** | **7.33 mph** | **0.7864** | **Selected Champion** |
| **Alternative: LightGBM Regressor** | 3.66 mph | 7.34 mph | 0.7861 | Candidate |

- **Hotspot Ranking Precision@10 on Severe Congestion:** **100.0%** (all 10 top-ranked hotspots in the held-out sample correspond to confirmed severe congestion events).

---

## 🖥️ Streamlit Dashboard Screens

1. **Screen 1: Overview (`pages/1_Overview.py`)**
   - Live KPI cards: 207 Monitored Stations, Latest Timestamp, Network Mean Speed, Congestion Distribution, High-Priority Count.
   - Interactive Mapbox map rendering all 207 stations with verified coordinates, colored by predicted condition.
   - Ranked Top 10 Hotspot Table with priority scores, speeds, corridors, and primary reasons.
   - Historical coverage banner prominently displaying March 1 – May 23, 2012.

2. **Screen 2: Hotspot Intelligence (`pages/2_Hotspot_Intelligence.py`)**
   - Three evidence-backed historical insight cards generated via strict factual template.
   - Hour-of-Day $\times$ Day-of-Week recurrence heatmap.
   - 24-hour diurnal speed curves with morning vs evening peak comparisons.
   - Top persistent bottleneck rankings ($\ge 15\%$ breakdown rate).
   - Sensor-specific deep dive diagnostics.

3. **Screen 3: Prediction Lab (`pages/3_Prediction_Lab.py`)**
   - What-If scenario simulator: Capacity reduction, duration, weather shocks.
   - Directed network queue propagation via `adj_mx.pkl`.
   - Dynamic weight sensitivity explorer: Live sliders to alter scoring weights and observe ranking changes in real-time without retraining.

4. **Screen 4: Action Center (`pages/4_Action_Center.py`)**
   - Decision-support intervention queue ranked 1 to 207.
   - Answers all 6 operational questions for selected bottlenecks.
   - Rule-based playbook triggers (Freeway Service Patrol, Ramp Metering, Dynamic Message Signs).
   - Instant CSV export of the intervention plan.

*(Supplementary pages: `pages/5_Model_Evaluation.py` for audit metrics and `pages/6_Data_Quality.py` for dataset validation reports).*

---

## 🛠️ Execution & Testing Commands

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Comprehensive Automated Tests
```powershell
python -m pytest tests/ -v
```
*(Runs 19 automated tests covering input validation, temporal leakage prevention, sensor boundaries, hotspot formula normalization, recurrence statistics, intervention answers, and pipeline execution).*

### 3. Run Chronological Backtest & Candidate Model Comparison
```powershell
python -m src.backtest_models
```

### 4. Evaluate Held-Out Models & Ranking Diagnostics
```powershell
python -m src.evaluate_model
```

### 5. Launch Streamlit Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Streamlit Community Cloud Deployment

1. **Repository Structure:** The repository includes a root `app.py` and `requirements.txt` delegating to `citypulse/`, allowing deployment whether the app entry point is configured as `app.py` or `citypulse/app.py`.
2. **Cloud Telemetry Cache:** The repository bundles verified lightweight snapshot and telemetry caches (`snapshot_cache.parquet`, `recent_telemetry_cache.parquet`, `traffic_patterns_cache.pkl` < 350 KB total) so the cloud dashboard operates out of the box with zero runtime errors.
3. **Deploying the Full 157 MB Training Dataset (Optional via Git LFS):**
   GitHub limits individual files to 100 MB. To track and push the full historical dataset `train.parquet` (157 MB) to Streamlit Cloud:
   ```bash
   git lfs install
   git lfs track "citypulse/data/train.parquet"
   git add .gitattributes citypulse/data/train.parquet
   git commit -m "Track full training dataset with Git LFS"
   git push origin main
   ```

