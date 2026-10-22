# CityPulse Future Machine Learning Plan (Phase 2 Roadmap)

> **Notice:** Machine-learning training has not started in Phase 1. Mumbai-specific modelling will begin only when suitable, verified Mumbai traffic observations and a clearly defined target variable are available.

---

## 1. Data Procurement Requirements for Phase 2

Before model training begins, the following empirical datasets must be secured:
1. **Physical Traffic Sensor Streams:** Continuous 5-minute to 15-minute average speed and flow measurements from Mumbai Municipal Corporation (MCGM) or Mumbai Traffic Police CCTV feeds.
2. **Road Network Graph Topology:** Graph adjacency matrix representing Mumbai's arterial corridors (Western Express Highway, Eastern Express Highway, SV Road, Link Road, Coastal Road) derived from OpenStreetMap geometry.
3. **Hyperlocal Weather Feeds:** Automated Weather Station (AWS) ward-level rain gauges from MCGM Disaster Management.

---

## 2. Planned Predictive Modules

### Module 1: Arterial Speed Prediction
- **Target Variable:** Segment Average Speed ($y_{t+\Delta t} \in \mathbb{R}^+$ in km/h for $\Delta t \in \{15, 30, 60\}$ minutes).
- **Candidate Architectures:**
  - Spatio-Temporal Graph Convolutional Networks (STGCN)
  - Diffusion Convolutional Recurrent Neural Networks (DCRNN)
  - LightGBM with spatial lags and time-of-day cyclic embeddings
- **Baseline Models:** Historical Average, Vector Autoregression (VAR), ARIMA
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

### Module 2: Junction Bottleneck Classification
- **Target Variable:** Level of Service (LOS) State Category ($y \in \{\text{LOS A/B (Free Flow)}, \text{LOS C/D (Moderate)}, \text{LOS E/F (Severe/Breakdown)}\}$).
- **Candidate Architectures:** XGBoost Classifier, Random Forest, Graph Attention Network (GAT)
- **Evaluation Metrics:** Macro F1-Score, Balanced Accuracy, Per-class Precision/Recall

### Module 3: Explainable AI (XAI) Attribution
- **Target Variable:** Feature Contribution Delta ($\Delta$ km/h to congestion onset)
- **Candidate Frameworks:** TreeSHAP, KernelSHAP, Integrated Gradients
- **Attribution Explanations:** Decompose model predictions into weather impact, upstream spillback, signal phase timing, and recurring historical baseline.

### Module 4: Model Evaluation & Concept Drift Monitoring
- **Monitoring Framework:** Rolling out-of-time evaluation, Population Stability Index (PSI), Wasserstein Distance on prediction residual distributions.
