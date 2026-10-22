# CityPulse AI Model Card: Highway Traffic Intelligence Pipeline

**Model Identifiers:** `traffic_classifier.joblib` & `traffic_forecaster.joblib`  
**Pipeline Version:** `v1.0.0-production`  
**Dataset:** METR-LA Highway Traffic Benchmark (Caltrans District 7)  
**Evaluation Date:** 2026-10-22 16:29:17 UTC  

---

## 1. Intended Use & Geographic Scope

- **Intended Domain:** Highway corridor decision support, congestion level classification, short-term speed forecasting, and priority bottleneck ranking.
- **Geographic Coverage:** **Los Angeles County Highway Network, California, USA** (207 inductive loop detector sensor stations across US-101, I-405, SR-134, I-110, I-5, I-210, and I-10 corridors).
- **Prohibited Uses:** Do NOT use for automated speed enforcement, punitive ticketing, or claims of accident causation.

---

## 2. Chronological Data Partitions (Zero Temporal Leakage)

- **Training Period:** March 1, 2012 to April 30, 2012 (3,634,299 available observations; 150,000 sampled for training).
- **Held-Out Test Period:** May 1, 2012 to May 23, 2012 (1,328,319 available observations; 50,000 sampled for test evaluation).
- **Zero Temporal Leakage Guarantee:** The test partition is strictly in the future relative to all training observations. Preprocessing scalers were fitted on the training split only.

---

## 3. Held-Out Test Evaluation Results

### A. Congestion Level Classification (15-Minute Horizon)
Target: Low (Speed >= 55 mph), Moderate (35 <= Speed < 55 mph), High (Speed < 35 mph).

| Model Candidate | Test Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 77.13% | 33.33% | 0.2903 | 0.6717 | Baseline |
| **Logistic Regression (L2)** | 87.01% | 65.93% | 0.6615 | 0.8391 | Linear Baseline |
| **HistGradientBoostingClassifier** | **90.00%** | **77.74%** | **0.7926** | **0.8962** | **Champion** |

### B. Speed Forecasting (15-Minute Horizon Continuous Speed)

| Model Candidate | Test MAE (mph) | Test RMSE (mph) | Test R² | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline (yt+3 = xt0)** | 3.74 mph | 8.08 mph | 0.7405 | Baseline |
| **Ridge Regression (L2)** | 3.80 mph | 7.49 mph | 0.7765 | Linear Baseline |
| **HistGradientBoostingRegressor** | **3.66 mph** | **7.33 mph** | **0.7864** | **Champion** |

---

## 4. Key Factors & Feature Importance

Top 3 drivers of congestion predictions:
1. `speed_current` (Mean importance: 0.1544)
2. `speed_mean_15m` (Mean importance: 0.0216)
3. `day_fraction` (Mean importance: 0.0063)

> **Scientific Caution:** Feature importance reflects predictive utility within the statistical model and does NOT establish physical or legal causation.
