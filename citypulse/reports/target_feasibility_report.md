# Target Feasibility Report: Model Capabilities Selection

**Platform:** CityPulse TrafficPulse AI  
**Dataset:** `data/train.parquet` (METR-LA Highway Traffic Benchmark)  

---

## 1. Capability Evaluation Matrix

| Capability Candidate | Target Definition | Feasibility | Scientific Justification & Decision |
| :--- | :--- | :---: | :--- |
| **Capability 1: Traffic Condition Classification** | Multi-class Congestion Level:<br>• **Low (Free-flow):** Speed >= 55 mph<br>• **Moderate:** 35 <= Speed < 55 mph<br>• **High (Congested):** Speed < 35 mph | **HIGHLY FEASIBLE**<br>**(SELECTED)** | Defensible highway Level of Service (LOS) standard. Grounded in actual speed measurements. Balanced representation across classes. Supports operational priority alerts. |
| **Capability 2: Multi-Horizon Speed Forecasting** | Continuous future speed:<br>• y_t+3 (15-min ahead)<br>• y_t+6 (30-min ahead)<br>• y_t+12 (60-min ahead) | **HIGHLY FEASIBLE**<br>**(SELECTED)** | Rich 12-step lookback sequence (X) directly aligned with 12 future horizons (Y). Direct evaluation against persistence baseline (MAE, RMSE, R^2). |
| **Capability 3: Explainable Location Prioritization** | Multi-factor Decision Ranking:<br>• Congestion Severity Index<br>• Speed Deficit from Free-flow<br>• Spatial Graph Centrality (`adj_mx`)<br>• Persistent Duration | **HIGHLY FEASIBLE**<br>**(SELECTED)** | Practical operational tool for traffic management centers. Avoids arbitrary accident-risk claims while ranking nodes by actionable congestion severity. |
| **Capability 4: Accident Risk Probability** | Probability of collision event | **UNSUPPORTED**<br>**(REJECTED)** | Dataset contains sensor speed loop measurements, not collision incident reports. Fabricating accident probabilities without ground truth would violate core integrity standards. |
| **Capability 5: Network Traffic Assignment** | Dynamic Origin-Destination Flow | **PARTIALLY SUPPORTED**<br>**(WHAT-IF ONLY)** | Supported through graph adjacency matrix `adj_mx` for local downstream shockwave propagation, but not citywide OD trip tables. |

---

## 2. Selected Target Definitions

### A. Classification Target: Congestion Level (C_t+k)
Defined at forecast horizon k = +3 (15 minutes ahead) and k = +6 (30 minutes ahead):
- **High Congestion (Red):** if y_t+k_d0 < 35.0 mph
- **Moderate Congestion (Amber):** if 35.0 <= y_t+k_d0 < 55.0 mph
- **Low / Free-Flow (Green):** if y_t+k_d0 >= 55.0 mph

### B. Regression Target: Future Highway Speed (y_t+3_d0)
Predict the exact speed in miles per hour 15 minutes into the future given the preceding 60 minutes of speed history and diurnal cycle.
