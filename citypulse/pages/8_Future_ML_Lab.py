import streamlit as st

st.markdown("""
<div class="citypulse-topbar">
    <div>
        <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">
            Future ML Lab — Predictive Architecture & Readiness Hub
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
            Architectural blueprint and data-readiness checklist for future machine-learning model development
        </p>
    </div>
    <div style="background: rgba(148, 163, 184, 0.15); border: 1px solid rgba(148, 163, 184, 0.4); border-radius: 8px; padding: 6px 12px; font-size: 0.75rem; color: #CBD5E1; font-weight: 700;">
        🧪 Phase 1 Interface Blueprint
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Disclaimer Banner
st.markdown("""
<div class="info-banner" style="border-left: 4px solid #06B6D4;">
    <strong style="color: #22D3EE; font-size: 0.95rem;">🔬 Phase 1 Machine Learning Training Status:</strong><br>
    <p style="margin: 6px 0; font-size: 0.88rem; font-weight: 600; color: #FFFFFF;">
        “Machine-learning training has not started. Mumbai-specific modelling will begin only when suitable, verified Mumbai traffic observations and a clearly defined target variable are available.”
    </p>
    <span style="font-size: 0.75rem; color: #94A3B8;">
        In strict adherence to the competition rules, no fabricated predictions, mock accuracy percentages, fake MAE/RMSE metrics, or synthetic performance curves are shown.
    </span>
</div>
""", unsafe_allow_html=True)

# Data Readiness Audit
st.subheader("📋 Mumbai Physical Sensor Readiness Checklist for ML")
st.caption("Prerequisites that must be fulfilled before Phase 2 model training can safely commence.")

r1, r2 = st.columns(2)
with r1:
    st.markdown("""
    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 14px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: #FFFFFF;">1. High-Frequency Physical Road Sensors</strong>
            <span style="background: rgba(239, 68, 68, 0.2); color: #EF4444; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">MISSING</span>
        </div>
        <p style="font-size: 0.75rem; color: #94A3B8; margin: 6px 0 0 0;">
            Requires calibrated physical road sensors (loop detectors, automated camera traffic counts) for Mumbai road segments. Current PDF contains annual aggregations.
        </p>
    </div>
    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: #FFFFFF;">2. Continuous Granular Timestamps</strong>
            <span style="background: rgba(245, 158, 11, 0.2); color: #F59E0B; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">PENDING</span>
        </div>
        <p style="font-size: 0.75rem; color: #94A3B8; margin: 6px 0 0 0;">
            Forecasting models require 5-minute to 15-minute sensor intervals. Annual enforcement totals cannot be used for minute-level speed forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)

with r2:
    st.markdown("""
    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 14px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: #FFFFFF;">3. Target Variable Ground Truth</strong>
            <span style="background: rgba(6, 182, 212, 0.2); color: #06B6D4; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">DEFINED (FUTURE)</span>
        </div>
        <p style="font-size: 0.75rem; color: #94A3B8; margin: 6px 0 0 0;">
            Target variable mathematically specified as Segment Average Speed (km/h) or Level of Service (LOS A through F).
        </p>
    </div>
    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: #FFFFFF;">4. Spatiotemporal Adjacency Matrix</strong>
            <span style="background: rgba(16, 185, 129, 0.2); color: #10B981; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">IN PREPARATION</span>
        </div>
        <p style="font-size: 0.75rem; color: #94A3B8; margin: 6px 0 0 0;">
            Mumbai arterial road network topology graph (excluding US PeMS layouts) to be generated from OpenStreetMap geometry.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Planned ML Modules Grid
st.subheader("🧪 Planned Predictive Modules (Phase 2 Architecture)")

modules = [
    {
        "id": "speed_prediction",
        "title": "Arterial Speed Prediction (Spatio-Temporal GNN)",
        "status": "Planned",
        "description": "Predict segment-level average vehicular speed 15, 30, and 60 minutes ahead across key corridors (e.g. Western Express Highway, Eastern Freeway).",
        "target_variable": "Segment Average Speed (km/h)",
        "input_features": ["Lagged speed (t-1...t-12)", "Historical speed profile", "Day of week", "Monsoon rainfall indicator", "Special events / VIP movement"],
        "split_plan": "Chronological split (70% train, 15% validation, 15% test) without temporal leakage",
        "baseline_model": "Historical Average & Vector Autoregression (VAR)",
        "candidate_models": ["DCRNN (Diffusion Convolutional RNN)", "Spatial-Temporal Graph Convolutional Network (STGCN)", "LightGBM regressor with spatial lags"],
        "evaluation_metrics": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)"],
        "limitations": "Model cannot generalize to unmonitored secondary roads; performance subject to sensor dropouts during intense monsoon storms."
    },
    {
        "id": "congestion_classification",
        "title": "Junction Congestion Severity Classifier",
        "status": "Planned",
        "description": "Multi-class classification of junction bottleneck states into standardized Level of Service tiers (Free Flow, Moderate, Severe, Breakdown).",
        "target_variable": "Level of Service Tier (LOS Category)",
        "input_features": ["Queue length estimate", "Junction approach speed", "Signal cycle duration", "Rainfall intensity (mm/h)"],
        "split_plan": "Stratified temporal splits across monsoon and non-monsoon test windows",
        "baseline_model": "Rule-based thresholding & Logistic Regression",
        "candidate_models": ["XGBoost Classifier", "Random Forest Ensemble", "Graph Attention Network (GAT)"],
        "evaluation_metrics": ["Macro F1-Score", "Balanced Accuracy", "Confusion Matrix per LOS Class"],
        "limitations": "High imbalance in 'Breakdown' state requires focal loss or oversampling techniques."
    },
    {
        "id": "explainable_ai",
        "title": "Explainable AI (XAI) Feature Attribution Hub",
        "status": "Planned",
        "description": "SHAP (SHapley Additive exPlanations) and TreeSHAP attribution to explain why a particular corridor is forecasted to enter severe congestion.",
        "target_variable": "Feature Contribution Delta (Δ km/h)",
        "input_features": ["All candidate model features"],
        "split_plan": "Applied across holdout test set anomalies",
        "baseline_model": "Linear regression coefficients",
        "candidate_models": ["TreeSHAP", "KernelSHAP", "Integrated Gradients for GNNs"],
        "evaluation_metrics": ["Attribution stability", "Monotonicity score", "Expert interpretability validation"],
        "limitations": "Attribution indicates correlational feature importance, not counterfactual physical causation."
    },
    {
        "id": "model_evaluation_benchmarking",
        "title": "Model Evaluation & Drift Monitoring",
        "status": "Planned",
        "description": "Continuous monitoring of concept drift, seasonal monsoon shifts, and post-enforcement behavioural changes.",
        "target_variable": "Model Residual Tracking (Error distribution)",
        "input_features": ["Prediction residuals over rolling 7-day windows"],
        "split_plan": "Rolling out-of-time evaluation",
        "baseline_model": "Fixed validation baseline",
        "candidate_models": ["Evidently AI / Population Stability Index (PSI)", "CUSUM change-point detection"],
        "evaluation_metrics": ["Wasserstein Distance", "Kolmogorov-Smirnov Test p-value"],
        "limitations": "Requires automated continuous sensor ingestion pipeline."
    }
]

m_col1, m_col2 = st.columns(2)
for idx, mod in enumerate(modules):
    target_col = m_col1 if idx % 2 == 0 else m_col2
    with target_col:
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.65rem; font-family: monospace; color: #06B6D4; background: #0B0F19; border: 1px solid #1E293B; padding: 2px 6px; border-radius: 4px;">{mod['id']}</span>
                <span style="font-size: 0.65rem; font-weight: 700; color: #F59E0B; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 6px; border-radius: 4px;">PLANNED</span>
            </div>
            <h3 style="margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; color: #FFFFFF;">{mod['title']}</h3>
            <p style="font-size: 0.75rem; color: #CBD5E1; margin: 0 0 12px 0; line-height: 1.4;">{mod['description']}</p>
            
            <div style="background: #0B0F19; border: 1px solid #1E293B; border-radius: 8px; padding: 10px; font-size: 0.72rem; line-height: 1.5;">
                <div><strong style="color: #06B6D4;">Target Variable:</strong> <span style="font-family: monospace; color: #F1F5F9;">{mod['target_variable']}</span></div>
                <div style="margin-top: 4px;"><strong style="color: #94A3B8;">Candidate Models:</strong> {', '.join(mod['candidate_models'])}</div>
                <div style="margin-top: 4px;"><strong style="color: #94A3B8;">Baseline:</strong> {mod['baseline_model']}</div>
                <div style="margin-top: 4px;"><strong style="color: #94A3B8;">Metrics:</strong> {', '.join(mod['evaluation_metrics'])}</div>
                <div style="margin-top: 4px;"><strong style="color: #94A3B8;">Validation:</strong> {mod['split_plan']}</div>
            </div>
            <div style="margin-top: 8px; font-size: 0.68rem; color: #EF4444;">
                <strong>Limitation:</strong> {mod['limitations']}
            </div>
        </div>
        """, unsafe_allow_html=True)
