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
from ml_utils import (
    parse_args, load_cohort, get_feature_matrix,
    make_output_path, experiment_name, print_run_banner,
    OUTPUT_DIR, TARGET
)
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
MLFLOW_DIR = Path('C:/tmp/vGFR_ML_v4')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
BASE_EXPERIMENT = "vGFR_Outlier_Study_V4"

# Use the R13 Champion features for this study
CHAMPION_FEATS = [
    'current_age', 'arterial_venecava_between_kidney_hepatic', 'E_late_right', 
    'venous_kidney_hu_std', 'E_arterial_right', 'venous_kidney_vol', 
    'late_right_kidney_vein', 'vol_per_age', 'arterial_portal_vein', 
    'age_x_E_arterial', 'E_arterial_mean', 'norm_ven_artery_right'
]

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
    args = parse_args("vGFR Outlier Study V4 (Round 15)")

    df, _ = load_cohort(args.cohort)
    X_full, y = get_feature_matrix(df, exclude_vol_hu=args.exclude_vol_hu)
    record_ids = df['record_id'].astype(str)
    print_run_banner("03_outlier_study_v4.py", args.cohort, df, X_full)

    # Use champion features that exist in this cohort's X
    champion_feats = [f for f in CHAMPION_FEATS if f in X_full.columns]
    missing = set(CHAMPION_FEATS) - set(champion_feats)
    if missing:
        print(f"  [warn] Champion features not available for this cohort: {missing}")

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(experiment_name(BASE_EXPERIMENT, args.cohort))

    # Filter scenario exclusion IDs to only those present in this cohort
    present_ids = set(record_ids.values)
    scenarios = [
        ("round_15_baseline", []),
        ("round_15_excl_19",   [i for i in ['19'] if any(i in r for r in present_ids)]),
        ("round_15_excl_18",   [i for i in ['18'] if any(i in r for r in present_ids)]),
        ("round_15_excl_both", [i for i in ['18', '19'] if any(i in r for r in present_ids)]),
    ]

    model = Ridge(alpha=10.0)
    pipe  = Pipeline([('scaler', StandardScaler()), ('model', model)])

    for label, exclude in scenarios:
        print(f"\n>>> Outlier Study: {label}")
        y_pred, metrics = evaluate_with_exclusions(
            X_full[champion_feats], y, record_ids, pipe, exclude
        )
        with mlflow.start_run(run_name=label):
            mlflow.log_params({'n_feats': len(champion_feats),
                               'excluded_ids': ','.join(exclude)})
            mlflow.log_metrics(metrics)

            plot_path = make_output_path(label, cohort=args.cohort)
            plot_egfrc_vs_vgfr(y.values, y_pred, label.replace('_', ' ').title(),
                               champion_feats, metrics, plot_path)
            mlflow.log_artifact(str(plot_path))
            print(f"[OK] {label}: MAE={metrics['MAE']:.2f} R2={metrics['R2']:.3f}")

if __name__ == '__main__':
    main()
