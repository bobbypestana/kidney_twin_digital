"""
vGFR Model Improvement Experiments -- Round 5 (MAE Optimization)

Experiment J: 
  Instead of adding features that maximize R2, the forward stepwise algorithm 
  is modified to add features that minimize MAE directly.
"""

import duckdb
import pandas as pd
import numpy as np
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from plot_egfrc_vs_vgfr import plot_egfrc_vs_vgfr
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import mlflow
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Config
# ============================================================================
DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data.duckdb'
OUTPUT_DIR = Path(__file__).parent / 'ml_results'
OUTPUT_DIR.mkdir(exist_ok=True)
MLFLOW_DIR = Path('C:/tmp/mlflow_vgfr')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "vGFR_Kidney_Digital_Twin"
TARGET = 'egfrc'

EXCLUDE_PREFIXES = ['vgfr_', 'conc_lit_', 'conc_late_', 'w_pv_', 'w_back_']
EXCLUDE_COLS = [
    'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
    'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days', TARGET,
]


# ============================================================================
# Core Functions
# ============================================================================
def load_and_prepare():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').fetchdf()
    conn.close()
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]

    drop_cols = list(EXCLUDE_COLS)
    for col in df.columns:
        for prefix in EXCLUDE_PREFIXES:
            if col.startswith(prefix):
                drop_cols.append(col)
                break
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_raw = df[feature_cols].select_dtypes(include=[np.number])
    y = df[TARGET]

    X = X_raw.copy()
    for phase, side in [('arterial','left'),('arterial','right'),
                        ('venous','left'),('venous','right'),
                        ('late','left'),('late','right')]:
        art, vein = f'{phase}_{side}_kidney_artery', f'{phase}_{side}_kidney_vein'
        if art in X.columns and vein in X.columns:
            X[f'E_{phase}_{side}'] = (X[art] - X[vein]) / X[art].replace(0, np.nan)

    for phase in ['arterial', 'venous', 'late']:
        la, ra = f'{phase}_left_kidney_artery', f'{phase}_right_kidney_artery'
        lv, rv = f'{phase}_left_kidney_vein', f'{phase}_right_kidney_vein'
        if la in X.columns and ra in X.columns:
            X[f'LR_ratio_artery_{phase}'] = X[la] / X[ra].replace(0, np.nan)
        if lv in X.columns and rv in X.columns:
            X[f'LR_ratio_vein_{phase}'] = X[lv] / X[rv].replace(0, np.nan)

    for side in ['left', 'right']:
        for v in ['artery', 'vein']:
            a, l = f'arterial_{side}_kidney_{v}', f'late_{side}_kidney_{v}'
            if a in X.columns and l in X.columns:
                X[f'phase_contrast_{v}_{side}'] = X[a] - X[l]

    for phase in ['arterial', 'venous', 'late']:
        aorta = f'{phase}_aorta'
        if aorta in X.columns:
            for side in ['left', 'right']:
                for v in ['artery', 'vein']:
                    col = f'{phase}_{side}_kidney_{v}'
                    if col in X.columns:
                        X[f'norm_{v}_{phase}_{side}'] = X[col] / X[aorta].replace(0, np.nan)

    for phase in ['arterial', 'venous', 'late']:
        la, ra = f'{phase}_left_kidney_artery', f'{phase}_right_kidney_artery'
        lv, rv = f'{phase}_left_kidney_vein', f'{phase}_right_kidney_vein'
        if la in X.columns and ra in X.columns:
            X[f'mean_artery_{phase}'] = (X[la] + X[ra]) / 2
        if lv in X.columns and rv in X.columns:
            X[f'mean_vein_{phase}'] = (X[lv] + X[rv]) / 2
        el, er = f'E_{phase}_left', f'E_{phase}_right'
        if el in X.columns and er in X.columns:
            X[f'E_{phase}_mean'] = (X[el] + X[er]) / 2

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    return X, y, data_hash


def evaluate_loocv(X_arr, y_arr, estimator):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }


def forward_stepwise_mae(X, y_arr, max_features=16, alpha=10.0):
    """Greedy forward selection focusing on minimizing MAE."""
    selected = []
    y_pred_best = None
    best_metrics = None
    best_mae_overall = np.inf

    remaining = [c for c in X.columns if c not in selected]

    while len(selected) < max_features and remaining:
        best_mae_step = np.inf
        best_feat_step = None
        best_pred_step = None
        best_metrics_step = None

        for feat in remaining:
            trial_feats = selected + [feat]
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            
            # Key difference: MINIMIZE MAE instead of maximizing R2
            if metrics['MAE'] < best_mae_step:
                best_mae_step = metrics['MAE']
                best_feat_step = feat
                best_pred_step = y_pred
                best_metrics_step = metrics

        # Only add if it improves overall MAE
        if best_mae_step < best_mae_overall:
            selected.append(best_feat_step)
            remaining.remove(best_feat_step)
            best_mae_overall = best_mae_step
            y_pred_best = best_pred_step
            best_metrics = best_metrics_step
        else:
            break

    return selected, y_pred_best, best_metrics


def log_run(name, tag, params, metrics, features, data_hash):
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tag('experiment_type', tag)
        mlflow.log_params({**params, 'data_hash': data_hash})
        mlflow.log_metrics(metrics)
        import json
        p = OUTPUT_DIR / f'info_{name}.json'
        with open(p, 'w') as f:
            json.dump({'features': features}, f, indent=2)
        mlflow.log_artifact(str(p))




# ============================================================================
# Experiments
# ============================================================================
def exp_J_mae_optim(X, y, data_hash):
    """J: Forward stepwise selection minimizing MAE instead of maximizing R2."""
    print(f"\n{'='*60}\nEXP J: Optimize Stepwise for MAE\n{'='*60}")
    
    results = {}
    
    # J1: Stepwise optimized for MAE
    feats_j1, y_pred_j1, metrics_j1 = forward_stepwise_mae(X, y.values, max_features=16, alpha=10.0)
    name = 'J1_Stepwise_Ridge_MAE_Optimized'
    log_run(name, 'mae_optimization', {'model': 'Ridge', 'selection_metric': 'MAE'}, metrics_j1, feats_j1, data_hash)
    results[name] = {
        'metrics': metrics_j1,
        'y_pred': y_pred_j1,
        'features': feats_j1,
        'model_name': 'Ridge'
    }
    
    print(f"  J1) Stepwise Ridge (MAE optimized, {len(feats_j1)} features): MAE={metrics_j1['MAE']:.2f}  R2={metrics_j1['R2']:.3f}")
    print(f"      Features: {feats_j1}")
    
    # J2: Same features but evaluated in BayesianRidge (like we did in R2/R4)
    est = Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())])
    y_pred_j2, metrics_j2 = evaluate_loocv(X[feats_j1].values, y.values, est)
    name_2 = 'J2_BayesianRidge_MAE_Features'
    log_run(name_2, 'mae_optimization', {'model': 'BayesianRidge', 'selection_metric': 'MAE'}, metrics_j2, feats_j1, data_hash)
    results[name_2] = {
        'metrics': metrics_j2,
        'y_pred': y_pred_j2,
        'features': feats_j1,
        'model_name': 'BayesianRidge'
    }
    
    print(f"  J2) BayesianRidge (same MAE-optim features): MAE={metrics_j2['MAE']:.2f}  R2={metrics_j2['R2']:.3f}")
    
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR Round 5 Improvements (MAE Optimization)")
    print("  Previous lowest MAE: Round 1 Stepwise Ridge (MAE=7.33, R2=0.533)")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, data_hash = load_and_prepare()
    print(f"[OK] {len(X)} patients, {len(X.columns)} features total")

    all_results = {}

    with mlflow.start_run(run_name=f"round5_mae_optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'round5_mae_optimization')
        mlflow.log_param('data_hash', data_hash)

        for k, v in exp_J_mae_optim(X, y, data_hash).items():
            all_results[k] = v

        # Unpack metrics for the dataframe
        results_metrics = {k: v['metrics'] for k, v in all_results.items()}
        results_df = pd.DataFrame(results_metrics).T.sort_values('MAE', ascending=True)
        # results_df.to_csv(OUTPUT_DIR / 'round5_results.csv')
        # mlflow.log_artifact(str(OUTPUT_DIR / 'round5_results.csv'))

        champion_name = results_df.index[0]
        champ_data = all_results[champion_name]
        champ_metrics = champ_data['metrics']
        champ_y_pred = champ_data['y_pred']
        champ_features = champ_data['features']
        model_label = f"R5 Champ (MAE) - {champ_data['model_name']}"

        # 1. Main Scatter Plot
        plot_path = OUTPUT_DIR / f'round5_champion_{champ_data["model_name"]}.png'
        plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, champ_features, champ_metrics, plot_path)
        mlflow.log_artifact(str(plot_path))

        # 2. Residual Plot
        res_path = OUTPUT_DIR / f'round5_champion_{champ_data["model_name"]}_residuals.png'
        residuals = y.values - champ_y_pred
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(champ_y_pred, residuals, s=100, alpha=0.7, edgecolors='black', linewidth=0.5, c='coral')
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel('Predicted vGFR'); ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title(f'Residual Plot: {model_label}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(res_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(res_path))
        print(f"[OK] Saved Champion Residuals: {res_path}")

if __name__ == '__main__':
    main()
