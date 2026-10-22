"""
5_Model_Evaluation.py - Model Performance, Validation & Explainability
Presents held-out test evaluation metrics, baseline comparisons, confusion matrices, and feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #FFFFFF;">
        Model Performance & Validation
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94A3B8;">
        Empirical evaluation results on the chronological held-out test set (May 1-23, 2012). Zero data leakage guarantee.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Load Evaluation Artifacts
reports_dir = "reports"
models_dir = "models"

try:
    with open(os.path.join(reports_dir, "final_evaluation.json"), "r") as f:
        eval_metrics = json.load(f)
    df_comp = pd.read_csv(os.path.join(reports_dir, "model_comparison.csv"))
    df_importance = pd.read_csv(os.path.join(reports_dir, "feature_importance.csv"))
except Exception as e:
    st.error(f"Error loading evaluation artifacts: {e}. Please run `python -m src.evaluate_model` first.")
    st.stop()

# 2. Key Test Metric Highlights
clf_eval = eval_metrics["classification"]
reg_eval = eval_metrics["regression"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Classification Test Accuracy", f"{clf_eval['test_accuracy']*100:.2f}%", f"+{(clf_eval['test_accuracy']-clf_eval['baseline_dummy_accuracy'])*100:.2f}% vs Dummy Baseline")
m2.metric("Classification Macro F1", f"{clf_eval['test_macro_f1']:.4f}", f"Balanced Acc: {clf_eval['test_balanced_accuracy']*100:.2f}%")
m3.metric("Forecasting Test MAE", f"{reg_eval['test_mae_mph']:.2f} mph", f"{(reg_eval['test_mae_mph']-reg_eval['baseline_persistence_mae']):.2f} mph vs Persistence", delta_color="inverse")
m4.metric("Forecasting Test R²", f"{reg_eval['test_r2']:.4f}", f"RMSE: {reg_eval['test_rmse_mph']:.2f} mph")

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 3. Model Comparison Table
st.markdown("### Candidate vs Baseline Model Benchmark")
st.caption("All models trained strictly on historical observations (< 2012-05-01) and evaluated on 50,000 held-out instances (May 1-23, 2012).")
st.dataframe(df_comp, use_container_width=True, hide_index=True)

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 4. Confusion Matrix and Feature Importance
col_cm, col_imp = st.columns([5, 5])

with col_cm:
    st.markdown("### Confusion Matrix (Classification Test)")
    cm_data = np.array(clf_eval["confusion_matrix"])
    labels = ["Low (>=55 mph)", "Mod (35-55 mph)", "High (<35 mph)"]
    
    fig_cm = px.imshow(
        cm_data,
        text_auto=True,
        labels=dict(x="Predicted Condition", y="True Condition", color="Instances"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8")
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_imp:
    st.markdown("### Permutation Feature Importance")
    fig_imp = px.bar(
        df_importance.head(8),
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        color="importance_mean",
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Mean Accuracy Degradation upon Permutation", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="", tickfont=dict(color="#94A3B8"), autorange="reversed"),
        showlegend=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# 5. Model Card Viewer
with st.expander("📄 View Formal Model Card (reports/model_card.md)"):
    model_card_path = os.path.join(reports_dir, "model_card.md")
    if os.path.exists(model_card_path):
        with open(model_card_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
