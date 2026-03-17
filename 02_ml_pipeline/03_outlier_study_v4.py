"""
vGFR Model Improvement Experiments -- Round 15 (Outlier Sensitivity)
Evaluates model stability when removing IDs 18 and 19 from training.
"""

import duckdb
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime
from plot_v2 import plot_egfrc_vs_vgfr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import mlflow
import warnings

warnings.filterwarnings('ignore')

# Config
DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data_v2.duckdb'
OUTPUT_DIR = Path(__file__).parent / 'ml_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DIR = Path('C:/tmp/vGFR_ML_v4')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "vGFR_Outlier_Study_V4"
TARGET = 'egfrc'

# Use the R13 Champion features for this study
CHAMPION_FEATS = [
    'current_age', 'arterial_venecava_between_kidney_hepatic', 'E_late_right', 
    'venous_kidney_hu_std', 'E_arterial_right', 'venous_kidney_vol', 
    'late_right_kidney_vein', 'vol_per_age', 'arterial_portal_vein', 
    'age_x_E_arterial', 'E_arterial_mean', 'norm_ven_artery_right'
]

def load_data():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.ml_features').fetchdf()
    conn.close()
    
    meta_cols = [
        'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
        'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days',
        'source_folder', TARGET
    ]
    X = df.drop(columns=[c for c in meta_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number]).astype(float).fillna(X.median()).fillna(0)
    y = df[TARGET]
    record_ids = df['record_id'].astype(str)
    return X, y, record_ids

def evaluate_with_exclusions(X, y, record_ids, estimator, exclude_from_train=[]):
    """
    LOOCV but with specific record_ids always excluded from training, but still predicted.
    """
    y_pred = np.zeros(len(y))
    loo = LeaveOneOut()
    
    X_arr = X.values
    y_arr = y.values
    
    for train_index, test_index in loo.split(X_arr):
        # Identify the record_ids in the training set
        train_record_ids = record_ids.iloc[train_index]
        
        # Filter out the manually excluded IDs from the training set
        mask = ~train_record_ids.astype(str).isin([str(x) for x in exclude_from_train])
        final_train_index = train_index[mask]
        
        # Train
        estimator.fit(X_arr[final_train_index], y_arr[final_train_index])
        
        # Predict the single hold-out
        y_pred[test_index] = estimator.predict(X_arr[test_index])
        
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }

def main():
    X, y, record_ids = load_data()
    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Scenarios: Baseline (none), No 19, No 18, No Both
    scenarios = [
        ("round_15_baseline", []),
        ("round_15_excl_19", ['19']),
        ("round_15_excl_18", ['18']),
        ("round_15_excl_both", ['18', '19'])
    ]

    model = Ridge(alpha=10.0)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])

    for label, exclude in scenarios:
        print(f"\n>>> Running Outlier Study: {label}")
        y_pred, metrics = evaluate_with_exclusions(X[CHAMPION_FEATS], y, record_ids, pipe, exclude)
        
        with mlflow.start_run(run_name=label):
            mlflow.log_params({'n_feats': len(CHAMPION_FEATS), 'excluded_ids': ','.join(exclude)})
            mlflow.log_metrics(metrics)
            
            plot_path = OUTPUT_DIR / f'{label}_champion.png'
            plot_egfrc_vs_vgfr(y.values, y_pred, label.replace('_', ' ').title(), CHAMPION_FEATS, metrics, plot_path)
            mlflow.log_artifact(str(plot_path))
            print(f"[OK] {label}: MAE={metrics['MAE']:.2f} R2={metrics['R2']:.3f}")

if __name__ == '__main__':
    main()
