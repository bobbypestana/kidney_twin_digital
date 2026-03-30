"""
vGFR Model Improvement Experiments -- Rounds 11 - 14
Focused on performance gains for the 25-patient cohort.
Standardized round names and plot output.
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
from sklearn.linear_model import Ridge, HuberRegressor, BayesianRidge
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import mlflow
import warnings

warnings.filterwarnings('ignore')

# Config
DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data_v2.duckdb'
MLFLOW_DIR = Path('C:/tmp/vGFR_ML_v3')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
BASE_EXPERIMENT = "vGFR_Improvements_V3"

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

def stepwise_blended_rank(X, y_arr, base_estimator, max_features=12):
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
            print(f"    Step {len(selected)}: Added '{best_row['feature']}' (Composite={score:.4f})")
        else:
            break
    return selected, y_pred_best, best_metrics

# ============================================================================
# Experiments
# ============================================================================

def exp_r11_hybrid_stats(X, y):
    print("\n>>> Round 11: Gold + Bronze Hybrid Stats (Rank-Blended)")
    model = BayesianRidge()
    feats, y_pred, metrics = stepwise_blended_rank(X, y.values, model)
    est = Pipeline([('scaler', StandardScaler()), ('model', model)])
    est.fit(X[feats], y)
    return feats, y_pred, metrics, "round_11", est

def exp_r12_stacking(X, y, selected_features):
    print("\n>>> Round 12: Stacking Ensemble (Bayesian + Huber + SVR)")
    estimators = [
        ('bayesian', BayesianRidge()),
        ('huber', HuberRegressor(max_iter=1000, epsilon=1.5)),
        ('svr', SVR(kernel='rbf', C=10.0))
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), passthrough=False)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', stack)])
    y_pred, metrics = evaluate_loocv(X[selected_features].values, y.values, pipe)
    pipe.fit(X[selected_features], y)
    return selected_features, y_pred, metrics, "round_12", pipe

def exp_r13_non_linear(X, y):
    print("\n>>> Round 13: Non-Linear Interaction Search")
    model = Ridge(alpha=10.0)
    feats, y_pred, metrics = stepwise_blended_rank(X, y.values, model)
    est = Pipeline([('scaler', StandardScaler()), ('model', model)])
    est.fit(X[feats], y)
    return feats, y_pred, metrics, "round_13", est

def exp_r14_bayesian_pruning(X, y):
    print("\n>>> Round 14: Bayesian Selecting based on Coefficient Stability")
    initial_feats, _, _ = stepwise_blended_rank(X, y.values, BayesianRidge(), max_features=15)
    
    model = BayesianRidge()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X[initial_feats])
    model.fit(X_s, y)
    
    stability = np.abs(model.coef_) / np.sqrt(np.diag(model.sigma_))
    stable_idx = np.argsort(stability)[-8:]
    final_feats = [initial_feats[i] for i in stable_idx]
    
    y_pred, metrics = evaluate_loocv(X[final_feats].values, y.values, Pipeline([('scaler', StandardScaler()), ('model', model)]))
    est = Pipeline([('scaler', StandardScaler()), ('model', model)])
    est.fit(X[final_feats], y)
    return final_feats, y_pred, metrics, "round_14", est

# ============================================================================
# Main Execution
# ============================================================================

def main():
    args = parse_args("vGFR Improvements V3 (Rounds 11–14)")

    df, data_hash = load_cohort(args.cohort)
    X, y = get_feature_matrix(df, exclude_vol_hu=args.exclude_vol_hu)
    print_run_banner("02_improvements_v3.py", args.cohort, df, X)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(experiment_name(BASE_EXPERIMENT, args.cohort))

    results = []

    f11, p11, m11, t11, e11 = exp_r11_hybrid_stats(X, y)
    results.append((f11, p11, m11, t11, e11))

    f13, p13, m13, t13, e13 = exp_r13_non_linear(X, y)
    results.append((f13, p13, m13, t13, e13))

    f12, p12, m12, t12, e12 = exp_r12_stacking(X, y, f13)
    results.append((f12, p12, m12, t12, e12))

    f14, p14, m14, t14, e14 = exp_r14_bayesian_pruning(X, y)
    results.append((f14, p14, m14, t14, e14))

    for feats, y_pred, metrics, tag, est in results:
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({'features': ', '.join(feats), 'n_feats': len(feats)})
            mlflow.log_metrics(metrics)

            plot_path = make_output_path(tag, cohort=args.cohort)
            plot_egfrc_vs_vgfr(y.values, y_pred, tag.replace('_', ' ').title(),
                               feats, metrics, plot_path)
            mlflow.log_artifact(str(plot_path))
            
            # [NEW] Systematic Export
            from ml_utils import export_champion_details
            export_champion_details(
                args.cohort, tag.replace('_', ' ').title(), "Champion",
                feats, est, metrics, y.values, y_pred
            )
            print(f"[OK] {tag}: MAE={metrics['MAE']:.2f} R2={metrics['R2']:.3f}")

if __name__ == '__main__':
    main()
