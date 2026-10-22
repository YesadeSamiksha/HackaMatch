"""
evaluate_model.py - Rigorous Evaluation on Chronological Held-Out Test Set (May 2012)
Evaluates trained models against baselines on unseen future observations.
Generates model_comparison.csv, final_evaluation.json, feature_importance.csv, and model_card.md.
"""

import os
import time
import json
import joblib
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report,
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)
from sklearn.inspection import permutation_importance

from src.preprocessing import (
    engineer_features, 
    create_targets, 
    FEATURE_COLS
)

def run_evaluation(test_sample_size: int = 50000, random_seed: int = 42):
    print("=" * 70)
    print(">>> CityPulse Traffic Intelligence: Held-Out Model Evaluation")
    print("=" * 70)
    
    start_time = time.time()
    parquet_path = os.path.join("data", "train.parquet")
    models_dir = "models"
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load artifacts
    print("[*] Loading trained model artifacts from models/...")
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessing_pipeline.joblib"))
    clf_hgb = joblib.load(os.path.join(models_dir, "traffic_classifier.joblib"))
    reg_hgb = joblib.load(os.path.join(models_dir, "traffic_forecaster.joblib"))
    clf_dummy = joblib.load(os.path.join(models_dir, "baseline_dummy_clf.joblib"))
    clf_logreg = joblib.load(os.path.join(models_dir, "baseline_logreg_clf.joblib"))
    reg_ridge = joblib.load(os.path.join(models_dir, "baseline_ridge_reg.joblib"))
    
    # 2. Load held-out test data (May 1 to May 23, 2012)
    split_date = "2012-05-01T00:00:00"
    print(f"[*] Reading held-out test partition (>= {split_date})...")
    
    cols_to_read = [
        "node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1",
        "y_t+3_d0" # Target: 15-min ahead speed
    ] + [f"x_t-{i}_d0" for i in range(1, 12)]
    
    table_test = pq.read_table(
        parquet_path, 
        columns=cols_to_read,
        filters=[("t0_timestamp", ">=", split_date)]
    )
    total_test_rows = len(table_test)
    print(f"[+] Total available held-out test observations: {total_test_rows:,}")
    
    np.random.seed(random_seed)
    if total_test_rows > test_sample_size:
        print(f"[*] Sampling {test_sample_size:,} held-out test instances for evaluation...")
        sample_indices = np.random.choice(total_test_rows, size=test_sample_size, replace=False)
        df_test_raw = table_test.take(sample_indices).to_pandas()
    else:
        df_test_raw = table_test.to_pandas()
        
    print("[*] Performing feature engineering on test partition...")
    X_test_raw = engineer_features(df_test_raw)
    y_test_clf, y_test_reg = create_targets(df_test_raw, horizon_step=3)
    
    # Transform test features using PRE-FITTED preprocessor (NEVER re-fit on test!)
    X_test_transformed = preprocessor.transform(X_test_raw)
    
    # -------------------------------------------------------------
    # 3. EVALUATE CLASSIFICATION MODELS
    # -------------------------------------------------------------
    print("\n--- Evaluating Classification Models on Held-Out Test Set ---")
    
    # Dummy Baseline
    y_pred_dummy = clf_dummy.predict(X_test_transformed)
    dummy_acc = accuracy_score(y_test_clf, y_pred_dummy)
    dummy_bal_acc = balanced_accuracy_score(y_test_clf, y_pred_dummy)
    dummy_f1_macro = f1_score(y_test_clf, y_pred_dummy, average="macro", zero_division=0)
    dummy_f1_weighted = f1_score(y_test_clf, y_pred_dummy, average="weighted", zero_division=0)
    
    # Logistic Regression Baseline
    y_pred_logreg = clf_logreg.predict(X_test_transformed)
    logreg_acc = accuracy_score(y_test_clf, y_pred_logreg)
    logreg_bal_acc = balanced_accuracy_score(y_test_clf, y_pred_logreg)
    logreg_f1_macro = f1_score(y_test_clf, y_pred_logreg, average="macro", zero_division=0)
    logreg_f1_weighted = f1_score(y_test_clf, y_pred_logreg, average="weighted", zero_division=0)
    
    # HistGradientBoosting Candidate
    y_pred_hgb = clf_hgb.predict(X_test_transformed)
    hgb_acc = accuracy_score(y_test_clf, y_pred_hgb)
    hgb_bal_acc = balanced_accuracy_score(y_test_clf, y_pred_hgb)
    hgb_f1_macro = f1_score(y_test_clf, y_pred_hgb, average="macro", zero_division=0)
    hgb_f1_weighted = f1_score(y_test_clf, y_pred_hgb, average="weighted", zero_division=0)
    hgb_cm = confusion_matrix(y_test_clf, y_pred_hgb)
    
    # Severe Congestion (Class 2) Specific Diagnostics
    # True Positives, False Positives, False Negatives for Class 2
    tp_high = hgb_cm[2, 2]
    fp_high = hgb_cm[0, 2] + hgb_cm[1, 2]
    fn_high = hgb_cm[2, 0] + hgb_cm[2, 1]
    recall_high = tp_high / (tp_high + fn_high) if (tp_high + fn_high) > 0 else 0.0
    prec_high = tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else 0.0
    f1_high = 2 * (prec_high * recall_high) / (prec_high + recall_high) if (prec_high + recall_high) > 0 else 0.0
    
    # Ranking Precision@10 on Severe Congestion:
    # Sort test slice by predicted lowest speeds and check true class 2 occurrence
    top10_idx = np.argsort(reg_hgb.predict(X_test_transformed))[:10]
    precision_at_10 = float((y_test_clf.iloc[top10_idx] == 2).mean())
    
    print(f"Dummy Baseline:        Accuracy={dummy_acc*100:.2f}%, Macro F1={dummy_f1_macro:.4f}")
    print(f"Logistic Regression:   Accuracy={logreg_acc*100:.2f}%, Macro F1={logreg_f1_macro:.4f}")
    print(f"HistGradientBoosting:  Accuracy={hgb_acc*100:.2f}%, Balanced Acc={hgb_bal_acc*100:.2f}%, Macro F1={hgb_f1_macro:.4f}")
    print(f"Severe Congestion (Class 2): Recall={recall_high*100:.1f}%, Precision={prec_high*100:.1f}%, F1={f1_high:.4f}")
    print(f"Hotspot Ranking Precision@10: {precision_at_10*100:.1f}%")
    print(f"False Positives (High): {fp_high}, False Negatives (High): {fn_high}")
    print("\nConfusion Matrix (Rows: True, Cols: Pred [0:Low, 1:Mod, 2:High]):\n", hgb_cm)
    
    # -------------------------------------------------------------
    # 4. EVALUATE REGRESSION MODELS
    # -------------------------------------------------------------
    print("\n--- Evaluating Speed Forecasting Models on Held-Out Test Set ---")
    
    # Persistence Baseline
    current_speed_test = X_test_raw["speed_current"].values
    persist_mae = mean_absolute_error(y_test_reg, current_speed_test)
    persist_rmse = np.sqrt(mean_squared_error(y_test_reg, current_speed_test))
    persist_r2 = r2_score(y_test_reg, current_speed_test)
    
    # Ridge Regression Baseline
    y_pred_ridge = reg_ridge.predict(X_test_transformed)
    ridge_mae = mean_absolute_error(y_test_reg, y_pred_ridge)
    ridge_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge))
    ridge_r2 = r2_score(y_test_reg, y_pred_ridge)
    
    # HistGradientBoosting Candidate
    y_pred_reg_hgb = reg_hgb.predict(X_test_transformed)
    hgb_reg_mae = mean_absolute_error(y_test_reg, y_pred_reg_hgb)
    hgb_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_hgb))
    hgb_reg_r2 = r2_score(y_test_reg, y_pred_reg_hgb)
    
    print(f"Persistence Baseline:  MAE={persist_mae:.2f} mph, RMSE={persist_rmse:.2f} mph, R²={persist_r2:.4f}")
    print(f"Ridge Regression:      MAE={ridge_mae:.2f} mph, RMSE={ridge_rmse:.2f} mph, R²={ridge_r2:.4f}")
    print(f"HistGradientBoosting:  MAE={hgb_reg_mae:.2f} mph, RMSE={hgb_reg_rmse:.2f} mph, R²={hgb_reg_r2:.4f}")
    
    # -------------------------------------------------------------
    # 5. FEATURE IMPORTANCE (Permutation Importance on Test Sample)
    # -------------------------------------------------------------
    print("\n[*] Computing Permutation Feature Importance on test slice (5,000 instances)...")
    perm_sample_idx = np.random.choice(len(X_test_transformed), size=5000, replace=False)
    X_perm = X_test_transformed[perm_sample_idx]
    y_perm_clf = y_test_clf.iloc[perm_sample_idx]
    
    perm_res = permutation_importance(
        clf_hgb, X_perm, y_perm_clf, 
        n_repeats=5, 
        random_state=random_seed, 
        n_jobs=1
    )
    
    feature_names = FEATURE_COLS
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_res.importances_mean,
        "importance_std": perm_res.importances_std
    }).sort_values(by="importance_mean", ascending=False)
    
    feat_imp_path = os.path.join(reports_dir, "feature_importance.csv")
    df_importance.to_csv(feat_imp_path, index=False)
    print(f"[+] Saved feature importance to {feat_imp_path}")
    print("\nTop 5 Influential Features:\n", df_importance.head(5))
    
    # -------------------------------------------------------------
    # 6. SAVE MODEL COMPARISON CSV
    # -------------------------------------------------------------
    model_comp_rows = [
        {
            "task": "Congestion Classification",
            "model": "Baseline: DummyClassifier (Most Frequent)",
            "primary_metric": f"Macro F1: {dummy_f1_macro:.4f}",
            "accuracy": f"{dummy_acc*100:.2f}%",
            "balanced_accuracy": f"{dummy_bal_acc*100:.2f}%",
            "macro_f1": f"{dummy_f1_macro:.4f}",
            "weighted_f1": f"{dummy_f1_weighted:.4f}",
            "mae_mph": "N/A",
            "rmse_mph": "N/A",
            "r2": "N/A",
            "status": "Baseline Reference"
        },
        {
            "task": "Congestion Classification",
            "model": "Baseline: Logistic Regression (L2)",
            "primary_metric": f"Macro F1: {logreg_f1_macro:.4f}",
            "accuracy": f"{logreg_acc*100:.2f}%",
            "balanced_accuracy": f"{logreg_bal_acc*100:.2f}%",
            "macro_f1": f"{logreg_f1_macro:.4f}",
            "weighted_f1": f"{logreg_f1_weighted:.4f}",
            "mae_mph": "N/A",
            "rmse_mph": "N/A",
            "r2": "N/A",
            "status": "Linear Baseline"
        },
        {
            "task": "Congestion Classification",
            "model": "Candidate: HistGradientBoostingClassifier",
            "primary_metric": f"Macro F1: {hgb_f1_macro:.4f}",
            "accuracy": f"{hgb_acc*100:.2f}%",
            "balanced_accuracy": f"{hgb_bal_acc*100:.2f}%",
            "macro_f1": f"{hgb_f1_macro:.4f}",
            "weighted_f1": f"{hgb_f1_weighted:.4f}",
            "mae_mph": "N/A",
            "rmse_mph": "N/A",
            "r2": "N/A",
            "status": "SELECTED CHAMPION"
        },
        {
            "task": "Speed Forecasting (t+15m)",
            "model": "Baseline: Persistence Model (yt+3 = xt0)",
            "primary_metric": f"MAE: {persist_mae:.2f} mph",
            "accuracy": "N/A",
            "balanced_accuracy": "N/A",
            "macro_f1": "N/A",
            "weighted_f1": "N/A",
            "mae_mph": f"{persist_mae:.2f}",
            "rmse_mph": f"{persist_rmse:.2f}",
            "r2": f"{persist_r2:.4f}",
            "status": "Domain Persistence Baseline"
        },
        {
            "task": "Speed Forecasting (t+15m)",
            "model": "Baseline: Ridge Regression (L2)",
            "primary_metric": f"MAE: {ridge_mae:.2f} mph",
            "accuracy": "N/A",
            "balanced_accuracy": "N/A",
            "macro_f1": "N/A",
            "weighted_f1": "N/A",
            "mae_mph": f"{ridge_mae:.2f}",
            "rmse_mph": f"{ridge_rmse:.2f}",
            "r2": f"{ridge_r2:.4f}",
            "status": "Linear Regularized Baseline"
        },
        {
            "task": "Speed Forecasting (t+15m)",
            "model": "Candidate: HistGradientBoostingRegressor",
            "primary_metric": f"MAE: {hgb_reg_mae:.2f} mph",
            "accuracy": "N/A",
            "balanced_accuracy": "N/A",
            "macro_f1": "N/A",
            "weighted_f1": "N/A",
            "mae_mph": f"{hgb_reg_mae:.2f}",
            "rmse_mph": f"{hgb_reg_rmse:.2f}",
            "r2": f"{hgb_reg_r2:.4f}",
            "status": "SELECTED CHAMPION"
        }
    ]
    df_comp = pd.DataFrame(model_comp_rows)
    comp_path = os.path.join(reports_dir, "model_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(f"[+] Saved model comparison to {comp_path}")
    
    # -------------------------------------------------------------
    # 7. SAVE FINAL EVALUATION JSON
    # -------------------------------------------------------------
    final_eval = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_partition": "May 1, 2012 to May 23, 2012 (Chronological Holdout)",
        "test_instances_evaluated": len(X_test_raw),
        "classification": {
            "model": "HistGradientBoostingClassifier",
            "test_accuracy": round(float(hgb_acc), 4),
            "test_balanced_accuracy": round(float(hgb_bal_acc), 4),
            "test_macro_f1": round(float(hgb_f1_macro), 4),
            "test_weighted_f1": round(float(hgb_f1_weighted), 4),
            "severe_congestion_recall": round(float(recall_high), 4),
            "severe_congestion_precision": round(float(prec_high), 4),
            "severe_congestion_f1": round(float(f1_high), 4),
            "ranking_precision_at_10": round(float(precision_at_10), 4),
            "false_positives_high": int(fp_high),
            "false_negatives_high": int(fn_high),
            "confusion_matrix": hgb_cm.tolist(),
            "baseline_dummy_accuracy": round(float(dummy_acc), 4),
            "baseline_logreg_accuracy": round(float(logreg_acc), 4),
            "baseline_logreg_macro_f1": round(float(logreg_f1_macro), 4)
        },
        "regression": {
            "model": "HistGradientBoostingRegressor",
            "test_mae_mph": round(float(hgb_reg_mae), 3),
            "test_rmse_mph": round(float(hgb_reg_rmse), 3),
            "test_r2": round(float(hgb_reg_r2), 4),
            "baseline_persistence_mae": round(float(persist_mae), 3),
            "baseline_persistence_rmse": round(float(persist_rmse), 3),
            "baseline_persistence_r2": round(float(persist_r2), 4),
            "baseline_ridge_mae": round(float(ridge_mae), 3),
            "baseline_ridge_rmse": round(float(ridge_rmse), 3),
            "baseline_ridge_r2": round(float(ridge_r2), 4)
        }
    }
    with open(os.path.join(reports_dir, "final_evaluation.json"), "w") as f:
        json.dump(final_eval, f, indent=2)
    print(f"[+] Saved final evaluation metrics to reports/final_evaluation.json")
    
    # -------------------------------------------------------------
    # 8. SAVE MODEL CARD (model_card.md)
    # -------------------------------------------------------------
    model_card = f"""# CityPulse AI Model Card: Highway Traffic Intelligence Pipeline

**Model Identifiers:** `traffic_classifier.joblib` & `traffic_forecaster.joblib`  
**Pipeline Version:** `v1.0.0-production`  
**Dataset:** METR-LA Highway Traffic Benchmark (Caltrans District 7)  
**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  

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
| **Dummy Baseline (Most Frequent)** | {dummy_acc*100:.2f}% | {dummy_bal_acc*100:.2f}% | {dummy_f1_macro:.4f} | {dummy_f1_weighted:.4f} | Baseline |
| **Logistic Regression (L2)** | {logreg_acc*100:.2f}% | {logreg_bal_acc*100:.2f}% | {logreg_f1_macro:.4f} | {logreg_f1_weighted:.4f} | Linear Baseline |
| **HistGradientBoostingClassifier** | **{hgb_acc*100:.2f}%** | **{hgb_bal_acc*100:.2f}%** | **{hgb_f1_macro:.4f}** | **{hgb_f1_weighted:.4f}** | **Champion** |

### B. Speed Forecasting (15-Minute Horizon Continuous Speed)

| Model Candidate | Test MAE (mph) | Test RMSE (mph) | Test R² | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline (yt+3 = xt0)** | {persist_mae:.2f} mph | {persist_rmse:.2f} mph | {persist_r2:.4f} | Baseline |
| **Ridge Regression (L2)** | {ridge_mae:.2f} mph | {ridge_rmse:.2f} mph | {ridge_r2:.4f} | Linear Baseline |
| **HistGradientBoostingRegressor** | **{hgb_reg_mae:.2f} mph** | **{hgb_reg_rmse:.2f} mph** | **{hgb_reg_r2:.4f}** | **Champion** |

---

## 4. Key Factors & Feature Importance

Top 3 drivers of congestion predictions:
1. `{df_importance.iloc[0]['feature']}` (Mean importance: {df_importance.iloc[0]['importance_mean']:.4f})
2. `{df_importance.iloc[1]['feature']}` (Mean importance: {df_importance.iloc[1]['importance_mean']:.4f})
3. `{df_importance.iloc[2]['feature']}` (Mean importance: {df_importance.iloc[2]['importance_mean']:.4f})

> **Scientific Caution:** Feature importance reflects predictive utility within the statistical model and does NOT establish physical or legal causation.
"""
    model_card_path = os.path.join(reports_dir, "model_card.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"[+] Saved model card to {model_card_path}")
    
    print(f"[SUCCESS] Model evaluation complete in {time.time()-start_time:.2f} seconds.")

if __name__ == "__main__":
    run_evaluation()
