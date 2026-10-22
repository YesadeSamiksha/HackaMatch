"""
train_model.py - Reproducible Training Pipeline for CityPulse TrafficPulse AI
Trains baseline and candidate models using chronological train split (< 2012-05-01).
Guarantees zero future data leakage.
"""

import os
import time
import json
import joblib
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import (
    engineer_features, 
    create_targets, 
    build_preprocessing_pipeline, 
    FEATURE_COLS, 
    NUMERICAL_COLS
)

def run_training(train_sample_size: int = 150000, random_seed: int = 42):
    print("=" * 70)
    print(">>> CityPulse Traffic Intelligence: Model Training Pipeline")
    print("=" * 70)
    
    start_total = time.time()
    parquet_path = os.path.join("data", "train.parquet")
    models_dir = "models"
    reports_dir = "reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    split_date = "2012-05-01T00:00:00"
    print(f"[*] Reading chronological training slice (< {split_date})...")
    
    cols_to_read = [
        "node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1",
        "y_t+3_d0" # Target: 15-minute ahead speed
    ] + [f"x_t-{i}_d0" for i in range(1, 12)]
    
    table_train = pq.read_table(
        parquet_path, 
        columns=cols_to_read,
        filters=[("t0_timestamp", "<", split_date)]
    )
    total_train_rows = len(table_train)
    print(f"[+] Total available training observations: {total_train_rows:,}")
    
    # Sample a representative stratified/chronological subset for efficient, reproducible training
    np.random.seed(random_seed)
    if total_train_rows > train_sample_size:
        print(f"[*] Sampling {train_sample_size:,} representative training instances...")
        sample_indices = np.random.choice(total_train_rows, size=train_sample_size, replace=False)
        df_train_raw = table_train.take(sample_indices).to_pandas()
    else:
        df_train_raw = table_train.to_pandas()
        
    print("[*] Performing feature engineering on training data...")
    X_train_raw = engineer_features(df_train_raw)
    y_train_clf, y_train_reg = create_targets(df_train_raw, horizon_step=3)
    
    print(f"[+] Feature matrix shape: {X_train_raw.shape}")
    print(f"[+] Classification target distribution:\n{y_train_clf.value_counts(normalize=True).round(3)}")
    
    # Fit preprocessing pipeline on training data ONLY
    print("[*] Fitting preprocessing transformations on training partition...")
    preprocessor = build_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    
    # -------------------------------------------------------------
    # 1. TRAIN CLASSIFICATION MODELS (Target: Congestion Level)
    # -------------------------------------------------------------
    print("\n--- [1/2] Training Congestion Classification Models ---")
    
    # Baseline 1: Dummy Classifier (Most Frequent)
    print("[*] Training Baseline 1: DummyClassifier (most_frequent)...")
    clf_dummy = DummyClassifier(strategy="most_frequent")
    clf_dummy.fit(X_train_transformed, y_train_clf)
    dummy_acc = accuracy_score(y_train_clf, clf_dummy.predict(X_train_transformed))
    print(f"    Baseline Dummy Train Accuracy: {dummy_acc*100:.2f}%")
    
    # Baseline 2: Logistic Regression
    print("[*] Training Baseline 2: LogisticRegression...")
    clf_logreg = LogisticRegression(max_iter=500, random_state=random_seed)
    clf_logreg.fit(X_train_transformed, y_train_clf)
    logreg_acc = accuracy_score(y_train_clf, clf_logreg.predict(X_train_transformed))
    logreg_f1 = f1_score(y_train_clf, clf_logreg.predict(X_train_transformed), average="macro")
    print(f"    Logistic Regression Train Acc: {logreg_acc*100:.2f}%, Macro F1: {logreg_f1:.4f}")
    
    # Candidate Model: HistGradientBoostingClassifier
    print("[*] Training Candidate Model: HistGradientBoostingClassifier...")
    t0 = time.time()
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=100, 
        learning_rate=0.1, 
        max_leaf_nodes=31, 
        random_state=random_seed
    )
    clf_hgb.fit(X_train_transformed, y_train_clf)
    hgb_acc = accuracy_score(y_train_clf, clf_hgb.predict(X_train_transformed))
    hgb_f1 = f1_score(y_train_clf, clf_hgb.predict(X_train_transformed), average="macro")
    print(f"    Candidate Gradient Boosting Train Acc: {hgb_acc*100:.2f}%, Macro F1: {hgb_f1:.4f} (took {time.time()-t0:.2f}s)")
    
    # -------------------------------------------------------------
    # 2. TRAIN REGRESSION MODELS (Target: Future Speed at t+15m)
    # -------------------------------------------------------------
    print("\n--- [2/2] Training Future Speed Forecasting Models ---")
    
    # Baseline 1: Persistence Model (Speed at t+15m = Current Speed at t0)
    current_speed_train = X_train_raw["speed_current"].values
    persist_mae = mean_absolute_error(y_train_reg, current_speed_train)
    persist_rmse = np.sqrt(mean_squared_error(y_train_reg, current_speed_train))
    persist_r2 = r2_score(y_train_reg, current_speed_train)
    print(f"    Baseline Persistence Train MAE: {persist_mae:.2f} mph, RMSE: {persist_rmse:.2f} mph, R²: {persist_r2:.4f}")
    
    # Baseline 2: Ridge Regression
    print("[*] Training Baseline 2: Ridge Regression...")
    reg_ridge = Ridge(alpha=1.0)
    reg_ridge.fit(X_train_transformed, y_train_reg)
    ridge_preds = reg_ridge.predict(X_train_transformed)
    ridge_mae = mean_absolute_error(y_train_reg, ridge_preds)
    ridge_r2 = r2_score(y_train_reg, ridge_preds)
    print(f"    Ridge Regression Train MAE: {ridge_mae:.2f} mph, R²: {ridge_r2:.4f}")
    
    # Candidate Model: HistGradientBoostingRegressor
    print("[*] Training Candidate Model: HistGradientBoostingRegressor...")
    t0 = time.time()
    reg_hgb = HistGradientBoostingRegressor(
        max_iter=100, 
        learning_rate=0.1, 
        max_leaf_nodes=31, 
        random_state=random_seed
    )
    reg_hgb.fit(X_train_transformed, y_train_reg)
    hgb_reg_preds = reg_hgb.predict(X_train_transformed)
    hgb_reg_mae = mean_absolute_error(y_train_reg, hgb_reg_preds)
    hgb_reg_rmse = np.sqrt(mean_squared_error(y_train_reg, hgb_reg_preds))
    hgb_reg_r2 = r2_score(y_train_reg, hgb_reg_preds)
    print(f"    Candidate Gradient Boosting Train MAE: {hgb_reg_mae:.2f} mph, RMSE: {hgb_reg_rmse:.2f} mph, R²: {hgb_reg_r2:.4f} (took {time.time()-t0:.2f}s)")
    
    # -------------------------------------------------------------
    # 3. SAVE ARTIFACTS
    # -------------------------------------------------------------
    print("\n[*] Saving model artifacts and metadata to models/...")
    
    # Save pipelines and models
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessing_pipeline.joblib"))
    joblib.dump(clf_hgb, os.path.join(models_dir, "traffic_classifier.joblib"))
    joblib.dump(reg_hgb, os.path.join(models_dir, "traffic_forecaster.joblib"))
    
    # Also save baselines for evaluation comparison
    joblib.dump(clf_dummy, os.path.join(models_dir, "baseline_dummy_clf.joblib"))
    joblib.dump(clf_logreg, os.path.join(models_dir, "baseline_logreg_clf.joblib"))
    joblib.dump(reg_ridge, os.path.join(models_dir, "baseline_ridge_reg.joblib"))
    
    # Save Feature Schema
    feature_schema = {
        "features": FEATURE_COLS,
        "numerical_features": NUMERICAL_COLS,
        "input_dimension": len(FEATURE_COLS),
        "target_classification": "congestion_level (0: Low, 1: Moderate, 2: High)",
        "target_regression": "speed_mph_t+15m (y_t+3_d0)",
        "horizon_minutes": 15
    }
    with open(os.path.join(models_dir, "feature_schema.json"), "w") as f:
        json.dump(feature_schema, f, indent=2)
        
    # Save Model Metadata
    metadata = {
        "model_version": "v1.0.0-production",
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset_name": "METR-LA Highway Traffic Benchmark (Caltrans District 7)",
        "geographic_scope": "Los Angeles County Highway Network (207 sensor stations)",
        "training_period": "2012-03-01 to 2012-04-30",
        "evaluation_holdout_period": "2012-05-01 to 2012-05-23",
        "total_train_available": total_train_rows,
        "train_sample_instances": len(X_train_raw),
        "split_strategy": "Strict Chronological Partition (Split Date: 2012-05-01)",
        "models": {
            "classification": {
                "algorithm": "HistGradientBoostingClassifier",
                "hyperparameters": {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
                "training_accuracy": round(float(hgb_acc), 4),
                "training_macro_f1": round(float(hgb_f1), 4)
            },
            "forecasting": {
                "algorithm": "HistGradientBoostingRegressor",
                "hyperparameters": {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
                "training_mae_mph": round(float(hgb_reg_mae), 3),
                "training_rmse_mph": round(float(hgb_reg_rmse), 3),
                "training_r2": round(float(hgb_reg_r2), 4)
            }
        }
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[SUCCESS] Model training complete in {time.time()-start_total:.2f} seconds. All artifacts saved to models/.")

if __name__ == "__main__":
    run_training()
