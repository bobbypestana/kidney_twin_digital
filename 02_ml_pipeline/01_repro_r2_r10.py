"""
Reproduction Suite for vGFR Experiments (Rounds 2 - 10)
Migrated to the v2 Pipeline architecture.
Starts from gold.ml_features.
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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
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
MLFLOW_DIR = Path('C:/tmp/vGFR_Repro_v2')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
BASE_EXPERIMENT = "vGFR_Repro_R2_R10"

# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.ml_features').fetchdf()
    conn.close()
    
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]
    
    # Identify meta columns
    meta_cols = [
        'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
        'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days',
        'source_folder', TARGET
    ]
    
    return df, meta_cols, data_hash

def get_base_Xy(df, meta_cols, filter_fn=None):
    X = df.drop(columns=[c for c in meta_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number]).astype(float).fillna(X.median()).fillna(0)
    
    if filter_fn:
        X = filter_fn(X)
        
    y = df[TARGET]
    return X, y

# ============================================================================
# Selection Strategies (Ported from Research Rounds)
# ============================================================================

def evaluate_loocv(X_arr, y_arr, estimator):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo, n_jobs=-1)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }

def stepwise_standard(X, y_arr, base_estimator, metric='R2', max_features=12):
    selected = []
    best_score_overall = -np.inf if metric == 'R2' else np.inf
    y_pred_best = None
    best_metrics = None
    remaining = list(X.columns)

    while len(selected) < max_features and remaining:
        best_score_step = -np.inf if metric == 'R2' else np.inf
        best_feat_step = None
        best_pred_step = None
        best_metrics_step = None

        print(f"    Step {len(selected)+1}: Evaluating {len(remaining)} candidates...", end="\r")
        for feat in remaining:
            trial_feats = selected + [feat]
            est = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            
            score = metrics[metric]
            if (metric == 'R2' and score > best_score_step) or (metric != 'R2' and score < best_score_step):
                best_score_step = score
                best_feat_step = feat
                best_pred_step = y_pred
                best_metrics_step = metrics

        if (metric == 'R2' and best_score_step > best_score_overall) or (metric != 'R2' and best_score_step < best_score_overall):
            selected.append(best_feat_step)
            remaining.remove(best_feat_step)
            best_score_overall = best_score_step
            y_pred_best = best_pred_step
            best_metrics = best_metrics_step
            print(f"    Step {len(selected)}: Added '{best_feat_step}' ({metric}={best_score_step:.4f})")
        else:
            print(f"    Stepwise converged at {len(selected)} features.                                ")
            break
    return selected, y_pred_best, best_metrics

def stepwise_blended_rank(X, y_arr, base_estimator, max_features=12):
    """Round 7: Rank-blended selection."""
    selected = []
    best_composite = np.inf
    y_pred_best = None
    best_metrics = None
    remaining = list(X.columns)

    while len(selected) < max_features and remaining:
        step_results = []
        print(f"    Step {len(selected)+1}: Ranking candidates...", end="\r")
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
        # Composite score for early stopping: (MAE/10 + RMSE/15 - R2)
        score = (m['MAE']/10.0) + (m['RMSE']/15.0) - m['R2']

        if score < best_composite:
            selected.append(best_row['feature'])
            remaining.remove(best_row['feature'])
            best_composite = score
            y_pred_best = best_row['pred']
            best_metrics = m
            print(f"    Step {len(selected)}: Added '{best_row['feature']}' (Composite={score:.4f})")
        else:
            print(f"    Blended Rank converged at {len(selected)} features.                     ")
            break
    return selected, y_pred_best, best_metrics

def stepwise_forced_n(X, y_arr, base_estimator, force_n=7, max_features=12, metric='MAE'):
    """Round 8: Forced N Features."""
    selected = []
    true_best_score = np.inf
    y_pred_best = None
    best_metrics = None
    remaining = list(X.columns)
    
    # Greedy pass to fill initial list
    temp_selected = []
    for step in range(max_features):
        best_score_step = np.inf
        best_feat_step = None
        
        for feat in remaining:
            trial = temp_selected + [feat]
            _, m = evaluate_loocv(X[trial].values, y_arr, Pipeline([('scaler', StandardScaler()), ('model', base_estimator)]))
            if m[metric] < best_score_step:
                best_score_step = m[metric]
                best_feat_step = feat
        
        if not best_feat_step: break
        
        temp_selected.append(best_feat_step)
        remaining.remove(best_feat_step)
        
        if best_score_step < true_best_score:
            true_best_score = best_score_step
            selected = list(temp_selected)
            # Re-evaluate to get pred/metrics
            y_pred_best, best_metrics = evaluate_loocv(X[selected].values, y_arr, Pipeline([('scaler', StandardScaler()), ('model', base_estimator)]))
        
        if len(temp_selected) >= force_n and best_score_step > true_best_score:
            break
            
    return selected, y_pred_best, best_metrics

def stepwise_pairwise(X, y_arr, base_estimator, max_features=12):
    """Round 4: Pairwise Bilateral."""
    selected = []
    best_r2 = -np.inf
    y_pred_best = None
    best_metrics = None
    all_cols = list(X.columns)

    def get_pair(f):
        if 'left' in f: 
            other = f.replace('left', 'right')
            return [f, other] if other in all_cols else [f]
        if 'right' in f:
            other = f.replace('right', 'left')
            return [f, other] if other in all_cols else [f]
        return [f]

    remaining_pairs = []
    seen = set()
    for c in all_cols:
        if c not in seen:
            p = tuple(sorted(get_pair(c)))
            remaining_pairs.append(p)
            for x in p: seen.add(x)

    while len(selected) < max_features and remaining_pairs:
        best_r2_step = -np.inf
        best_pair_step = None
        
        for pair in remaining_pairs:
            trial = selected + list(pair)
            if len(trial) > max_features: continue
            _, m = evaluate_loocv(X[trial].values, y_arr, Pipeline([('scaler', StandardScaler()), ('model', base_estimator)]))
            if m['R2'] > best_r2_step:
                best_r2_step = m['R2']
                best_pair_step = pair
                best_pred_step = _
                best_metrics_step = m

        if best_r2_step > best_r2:
            selected.extend(list(best_pair_step))
            remaining_pairs.remove(best_pair_step)
            best_r2 = best_r2_step
            y_pred_best, best_metrics = evaluate_loocv(X[selected].values, y_arr, Pipeline([('scaler', StandardScaler()), ('model', base_estimator)]))
        else:
            break
    return selected, y_pred_best, best_metrics

# ============================================================================
# Main Reproduction Runner
# ============================================================================

def run_experiment(round_id, X, y, data_hash, strategy_fn, models, params):
    print(f"\n>>> Running Round {round_id} reproduction...")
    results = {}
    
    with mlflow.start_run(run_name=f"Round_{round_id}") as parent:
        mlflow.log_params({**params, 'data_hash': data_hash})
        
        for model_name, estimator in models.items():
            print(f"  Testing {model_name}...")
            feats, y_pred, metrics = strategy_fn(X, y.values, estimator)
            
            with mlflow.start_run(run_name=f"R{round_id}_{model_name}", nested=True):
                mlflow.log_params({'model': model_name, 'n_features': len(feats)})
                mlflow.log_metrics(metrics)
                mlflow.log_param('features', ', '.join(feats))
                
                # Fit final on selected features to capture weights
                final_est = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
                final_est.fit(X[feats], y)
                
                results[model_name] = {
                    'metrics': metrics, 
                    'y_pred': y_pred, 
                    'features': feats,
                    'estimator': final_est
                }
                print(f"    MAE={metrics['MAE']:.2f} R2={metrics['R2']:.3f} Features={len(feats)}")

        # Champion for the round
        best_model = max(results, key=lambda k: results[k]['metrics']['R2'])
        best = results[best_model]
        plot_path = make_output_path(f'round_{round_id}', cohort=_COHORT)
        plot_egfrc_vs_vgfr(y.values, best['y_pred'], f"Round {round_id} Champ: {best_model}",
                           best['features'], best['metrics'], plot_path)
        mlflow.log_artifact(str(plot_path))
        
        # [NEW] Systematic Export (using the actual fitted estimator from the trial)
        from ml_utils import export_champion_details
        export_champion_details(
            _COHORT, f"Round {round_id}", best_model,
            best['features'], best['estimator'], best['metrics'],
            y.values, best['y_pred'],
            is_report_champion=(str(round_id) == "3")
        )
        
    return results

_COHORT = None  # set in main

def main():
    global _COHORT
    args = parse_args("vGFR Reproduction Suite (Rounds 2–10)")
    _COHORT = args.cohort

    df, data_hash = load_cohort(_COHORT)
    X, y = get_feature_matrix(df, exclude_vol_hu=args.exclude_vol_hu)
    print_run_banner("01_repro_r2_r10.py", _COHORT, df, X)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(experiment_name(BASE_EXPERIMENT, _COHORT))

    # ── Filter helpers that operate on X (no meta needed) ────────────────────
    def filter_kidney(X_):
        return X_[[c for c in X_.columns if 'kidney' in c.lower() or 'aorta' in c.lower()]]

    # Models
    ridge    = Ridge(alpha=10.0)
    bayesian = BayesianRidge()
    rf       = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42)
    svr      = SVR(kernel='rbf', C=10.0)

    run_experiment("2",  X, y, data_hash, stepwise_standard,
                   {'Ridge': ridge, 'Bayesian': bayesian}, {'type': 'Baseline Stepwise'})
    run_experiment("3",  filter_kidney(X), y, data_hash, stepwise_standard,
                   {'Ridge': ridge}, {'restriction': 'Kidney-Only'})
    run_experiment("4",  X, y, data_hash, stepwise_pairwise,
                   {'Ridge': ridge, 'Bayesian': bayesian}, {'type': 'Pairwise'})
    run_experiment("5",  X, y, data_hash,
                   lambda X_, y_, est: stepwise_standard(X_, y_, est, metric='MAE'),
                   {'Ridge': ridge}, {'type': 'MAE-Optim'})
    run_experiment("7",  X, y, data_hash, stepwise_blended_rank,
                   {'Ridge': ridge, 'Bayesian': bayesian, 'RF': rf}, {'type': 'Rank-Blended'})
    run_experiment("8",  X, y, data_hash,
                   lambda X_, y_, est: stepwise_forced_n(X_, y_, est, force_n=7),
                   {'Ridge': ridge, 'Bayesian': bayesian}, {'type': 'Forced-N'})
    run_experiment("10", X, y, data_hash, stepwise_blended_rank,
                   {'Ridge': ridge, 'Bayesian': bayesian, 'RF': rf,
                    'SVR': svr, 'Huber': HuberRegressor()}, {'type': 'Full-Competition'})

    print(f"\n[DONE] Tracking logged to: {MLFLOW_DIR}")

if __name__ == '__main__':
    main()
