"""
vGFR Model Improvement Experiments -- Round 4 (Explainability)

Experiment I: Pairwise Bilateral Features
  For clinical explainability, if a 'left' kidney feature is used, its 'right' 
  counterpart MUST also be included, and vice versa. 
  This script implements a custom forward stepwise selection that adds 
  features in bilateral pairs.
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
    # Add extraction ratios
    for phase, side in [('arterial','left'),('arterial','right'),
                        ('venous','left'),('venous','right'),
                        ('late','left'),('late','right')]:
        art, vein = f'{phase}_{side}_kidney_artery', f'{phase}_{side}_kidney_vein'
        if art in X.columns and vein in X.columns:
            X[f'E_{phase}_{side}'] = (X[art] - X[vein]) / X[art].replace(0, np.nan)

    # Note: We omit the _mean and _ratio features here because we want the model 
    # to explicitly use the left and right raw features as pairs.
    for phase in ['arterial', 'venous', 'late']:
        aorta = f'{phase}_aorta'
        if aorta in X.columns:
            for side in ['left', 'right']:
                for v in ['artery', 'vein']:
                    col = f'{phase}_{side}_kidney_{v}'
                    if col in X.columns:
                        X[f'norm_{v}_{phase}_{side}'] = X[col] / X[aorta].replace(0, np.nan)

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


def get_feature_pair(feat, all_cols):
    """If feat is left/right specific, return [feat, counterpart]. Otherwise return [feat]."""
    if 'left' in feat:
        counterpart = feat.replace('left', 'right')
        if counterpart in all_cols:
            return [feat, counterpart]
    elif 'right' in feat:
        counterpart = feat.replace('right', 'left')
        if counterpart in all_cols:
            return [feat, counterpart]
    return [feat]


def forward_stepwise_pairwise(X, y_arr, max_features=16, alpha=10.0, base_features=None):
    """Greedy forward selection that adds features in left/right pairs."""
    if base_features is None:
        base_features = []
    
    selected = list(base_features)
    y_pred_best = None
    best_metrics = None
    
    if selected:
        est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
        y_pred_best, best_metrics = evaluate_loocv(X[selected].values, y_arr, est)
        best_r2_overall = best_metrics['R2']
    else:
        best_r2_overall = -np.inf

    # Group remaining features into pairs (or singletons for systemic features/age)
    remaining_pairs = []
    seen = set(selected)
    for c in X.columns:
        if c not in seen:
            pair = tuple(sorted(get_feature_pair(c, X.columns)))
            if pair not in remaining_pairs:
                remaining_pairs.append(pair)

    while len(selected) < max_features and remaining_pairs:
        best_r2_step = -np.inf
        best_pair_step = None
        best_pred_step = None
        best_metrics_step = None

        for pair in remaining_pairs:
            trial_feats = selected + list(pair)
            
            # Skip if adding this pair exceeds max_features
            if len(trial_feats) > max_features:
                continue
                
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            
            if metrics['R2'] > best_r2_step:
                best_r2_step = metrics['R2']
                best_pair_step = pair
                best_pred_step = y_pred
                best_metrics_step = metrics

        if best_pair_step is None:
            break

        # Only add if it improves overall R2
        if best_r2_step > best_r2_overall:
            selected.extend(list(best_pair_step))
            remaining_pairs.remove(best_pair_step)
            best_r2_overall = best_r2_step
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
        # info = {'features': features}
        # p = OUTPUT_DIR / f'info_{name}.json'
        # with open(p, 'w') as f:
        #     json.dump(info, f, indent=2, default=str)
        # mlflow.log_artifact(str(p))




# ============================================================================
# Experiments
# ============================================================================
def exp_I_pairwise(X, y, data_hash):
    """I: Forward stepwise selection forcing left/right pairs."""
    print(f"\n{'='*60}\nEXP I: Explainable Pairwise Bilateral Model\n{'='*60}")
    
    results = {}
    
    # I1: Pairwise Stepwise (Ridge)
    feats_i1, y_pred_i1, metrics_i1 = forward_stepwise_pairwise(X, y.values, max_features=16, alpha=10.0)
    name = 'I1_Pairwise_Stepwise_Ridge'
    log_run(name, 'explainable', {'model': 'Ridge_Pairwise'}, metrics_i1, feats_i1, data_hash)
    results[name] = {
        'metrics': metrics_i1,
        'y_pred': y_pred_i1,
        'features': feats_i1,
        'model_name': 'Ridge_Pairwise'
    }
    
    print(f"  I1) Pairwise Ridge (Stepwise, {len(feats_i1)} features): R2={metrics_i1['R2']:.3f}  MAE={metrics_i1['MAE']:.2f}")
    print(f"      Features: {feats_i1}")
    
    # I2: Pairwise Stepwise (BayesianRidge evaluation)
    # Use features selected by I1, evaluate in BayesianRidge since it was globally best
    est = Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())])
    y_pred_i2, metrics_i2 = evaluate_loocv(X[feats_i1].values, y.values, est)
    name_2 = 'I2_Pairwise_BayesianRidge'
    log_run(name_2, 'explainable', {'model': 'BayesianRidge_Pairwise'}, metrics_i2, feats_i1, data_hash)
    results[name_2] = {
        'metrics': metrics_i2,
        'y_pred': y_pred_i2,
        'features': feats_i1,
        'model_name': 'BayesianRidge_Pairwise'
    }
    
    print(f"  I2) Pairwise BayesianRidge (same {len(feats_i1)} features): R2={metrics_i2['R2']:.3f}  MAE={metrics_i2['MAE']:.2f}")
    
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR Round 4 Improvements (Explainable Bilateral Pairs)")
    print("  Current global best: BayesianRidge Stepwise R2=0.574, MAE=7.53")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, data_hash = load_and_prepare()
    print(f"[OK] {len(X)} patients, {len(X.columns)} features total")

    all_results = {}

    with mlflow.start_run(run_name=f"round4_explainable_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'round4_explainability')
        mlflow.log_param('data_hash', data_hash)

        for k, v in exp_I_pairwise(X, y, data_hash).items():
            all_results[k] = v

        # Unpack metrics for the dataframe
        results_metrics = {k: v['metrics'] for k, v in all_results.items()}
        results_df = pd.DataFrame(results_metrics).T.sort_values('R2', ascending=False)
        results_df.to_csv(OUTPUT_DIR / 'round4_results.csv')
        mlflow.log_artifact(str(OUTPUT_DIR / 'round4_results.csv'))

        champion_name = results_df.index[0]
        champ_data = all_results[champion_name]
        champ_metrics = champ_data['metrics']
        champ_y_pred = champ_data['y_pred']
        champ_features = champ_data['features']
        model_label = f"R4 Champ (Bilateral Pairs) - {champ_data['model_name']}"

        # 1. Main Scatter Plot
        plot_path = OUTPUT_DIR / f'round4_champion_{champ_data["model_name"]}.png'
        plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, champ_features, champ_metrics, plot_path)
        mlflow.log_artifact(str(plot_path))

        # 2. Residual Plot
        res_path = OUTPUT_DIR / f'round4_champion_{champ_data["model_name"]}_residuals.png'
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
