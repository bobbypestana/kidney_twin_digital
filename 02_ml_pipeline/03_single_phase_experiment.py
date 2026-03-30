"""
vGFR Phase-Specific ML Experiments
Loops through purely 'arterial', 'venous', or 'late' feature sets.
Runs rank-blended selection for Ridge, BayesianRidge, Huber, RF, GB, ElasticNet, KNN, XGBoost.
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, HuberRegressor, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import mlflow

from ml_utils import (
    parse_args, load_cohort, get_feature_matrix,
    make_output_path, experiment_name, print_run_banner,
    export_champion_details, OUTPUT_DIR
)
from plot_v2 import plot_egfrc_vs_vgfr

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MLFLOW_DIR = Path('C:/tmp/vGFR_ML_Phases')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
BASE_EXPERIMENT = "vGFR_Single_Phase"

# ============================================================================
# Feature Engineering & Filtering
# ============================================================================

def derive_intra_phase_features(X):
    """Engineer new spatial delta features within the same phase."""
    df = X.copy()
    
    # Arterial Asymmetry
    if 'arterial_left_kidney_artery' in df.columns and 'arterial_right_kidney_artery' in df.columns:
        df['arterial_k_artery_asymmetry'] = (df['arterial_left_kidney_artery'] - df['arterial_right_kidney_artery']).abs()
        
    # Venous Organ-to-Systemic Gradient
    if 'venous_left_kidney_vein' in df.columns and 'venous_venacava_above_hepatic' in df.columns:
        df['venous_k_vein_systemic_grad'] = (df['venous_left_kidney_vein'] - df['venous_venacava_above_hepatic']).abs()

    # Late Portal/Hepatic Delta
    if 'late_portal_vein' in df.columns and 'late_left_hepatic_vein' in df.columns:
        df['late_portal_hepatic_delta'] = (df['late_portal_vein'] - df['late_left_hepatic_vein'])
        
    return df

def get_phase_features(all_cols, target_phase):
    """Filters columns to only include true global demographics and the target phase."""
    globals_true = ['current_age', 'sex']
    selected = [c for c in globals_true if c in all_cols]
    
    for c in all_cols:
        if c in globals_true: continue
        
        # Categorize
        if any(substring in c for substring in ['art_flow_efficiency', 'mean_artery_arterial', 'arterial', 'norm_art_', 'age_x_E_arterial', 'E_arterial']):
            feat_phase = 'arterial'
        elif any(substring in c for substring in ['ven_flow_efficiency', 'mean_artery_venous', 'venous', 'norm_ven_', 'E_venous']):
            feat_phase = 'venous'
        elif any(substring in c for substring in ['late', 'E_late']):
            feat_phase = 'late'
        else:
            feat_phase = 'unknown' # Things like 'kidney_vol' should be gone, but catch-all
            
        if feat_phase == target_phase:
            selected.append(c)
            
    return selected

# ============================================================================
# Core LOOCV Engine
# ============================================================================

def evaluate_loocv(X_arr, y_arr, estimator):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo, n_jobs=-1)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }

def stepwise_blended_rank(X, y_arr, base_estimator, max_features=10):
    selected = []
    best_composite = np.inf
    y_pred_best = None
    best_metrics = None
    remaining = list(X.columns)

    while len(selected) < max_features and remaining:
        step_results = []
        for feat in remaining:
            trial_feats = selected + [feat]
            est = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            step_results.append({'feature': feat, 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE'], 'R2': metrics['R2'], 'metrics': metrics, 'pred': y_pred})

        df_step = pd.DataFrame(step_results)
        df_step['Rank_MAE'] = df_step['MAE'].rank(ascending=True)
        df_step['Rank_RMSE'] = df_step['RMSE'].rank(ascending=True)
        df_step['Rank_R2'] = df_step['R2'].rank(ascending=False)
        df_step['Avg_Rank'] = df_step[['Rank_MAE', 'Rank_RMSE', 'Rank_R2']].mean(axis=1)

        best_row = df_step.sort_values('Avg_Rank').iloc[0]
        m = best_row['metrics']
        score = (m['MAE']/10.0) + (m['RMSE']/15.0) - m['R2']

        if score < best_composite:
            selected.append(best_row['feature'])
            remaining.remove(best_row['feature'])
            best_composite = score
            y_pred_best = best_row['pred']
            best_metrics = m
        else:
            break
            
    return selected, y_pred_best, best_metrics

# ============================================================================
# Main Execution
# ============================================================================

def main():
    args = parse_args("vGFR Single-Phase Experiments")

    df, data_hash = load_cohort(args.cohort)
    X_raw, y = get_feature_matrix(df, exclude_vol_hu=args.exclude_vol_hu, exclude_age=args.exclude_age)
    X = derive_intra_phase_features(X_raw)
    
    print_run_banner("03_single_phase_experiment.py", args.cohort, df, X)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(experiment_name(BASE_EXPERIMENT, args.cohort))

    estimators = [
        ("Ridge", Ridge(alpha=1.0)),
        ("BayesianRidge", BayesianRidge()),
        ("Huber", HuberRegressor(max_iter=1000, epsilon=1.35)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("KNN", KNeighborsRegressor(n_neighbors=5, weights='distance'))
    ]
    
    if HAS_XGB:
        estimators.append(("XGBoost", XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')))

    phases = ['arterial', 'venous', 'late']
    
    for phase in phases:
        print(f"\n{'='*60}")
        print(f"=== EVALUATING PHASE: {phase.upper()} ===")
        print(f"{'='*60}")
        
        phase_cols = get_phase_features(X.columns, phase)
        print(f"Features available for {phase}: {len(phase_cols)}")
        X_phase = X[phase_cols]
        
        phase_results = []
        
        for model_name, model in estimators:
            print(f"\n>>> Running {model_name}...")
            feats, y_pred, metrics = stepwise_blended_rank(X_phase, y.values, model, max_features=8)
            
            if metrics is not None:
                print(f"    Selected {len(feats)} features: MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.3f}")
                est = Pipeline([('scaler', StandardScaler()), ('model', model)])
                est.fit(X_phase[feats], y)
                phase_results.append({
                    'model_name': model_name,
                    'feats': feats,
                    'y_pred': y_pred,
                    'metrics': metrics,
                    'model': est
                })
            else:
                print(f"    Failed to select features.")

        # Rank Top 3 for this phase based on R2 and MAE
        if phase_results:
            phase_results.sort(key=lambda x: (x['metrics']['MAE'] / 10.0) - x['metrics']['R2'])
            top_3 = phase_results[:3]
            
            for i, result in enumerate(top_3):
                rank = i + 1
                prefix = f"{phase}_no_age" if args.exclude_age else phase
                tag = f"{prefix}_top{rank}_{result['model_name']}"
                
                with mlflow.start_run(run_name=tag):
                    mlflow.log_params({'phase': phase, 'model': result['model_name'], 'features': ', '.join(result['feats'])})
                    mlflow.log_metrics(result['metrics'])

                    plot_path = make_output_path(tag, cohort=args.cohort)
                    plot_egfrc_vs_vgfr(y.values, result['y_pred'], f"{phase.title()} Phase - Rank {rank} - {result['model_name']}",
                                       result['feats'], result['metrics'], plot_path)
                    
                    export_name = f"{prefix}_rank{rank}_{result['model_name'][:8]}"
                    export_champion_details(
                        args.cohort, export_name, result['model_name'],
                        result['feats'], result['model'], result['metrics'], y.values, result['y_pred']
                    )

if __name__ == '__main__':
    main()
