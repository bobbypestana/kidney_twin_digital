"""
vGFR Model Improvement Experiments -- Round 3 (Feature Restrictions)

Experiments:
  F: Remove Age - Forward stepwise selection without 'current_age'.
  G: Bilateral Mean - Stepwise selection restricted to _mean, _ratio, and systemic vessels.
  H: Strict Kidney-Only - Stepwise selection without age, without non-kidney/non-aorta vessels.
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


def forward_stepwise(X, y_arr, max_features=15, alpha=10.0, base_features=None):
    """Greedy forward selection."""
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

    remaining = [c for c in X.columns if c not in selected]

    while len(selected) < max_features and remaining:
        best_r2_step = -np.inf
        best_feat_step = None
        best_pred_step = None
        best_metrics_step = None

        for feat in remaining:
            trial_feats = selected + [feat]
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            
            if metrics['R2'] > best_r2_step:
                best_r2_step = metrics['R2']
                best_feat_step = feat
                best_pred_step = y_pred
                best_metrics_step = metrics

        # Only add if it improves overall R2
        if best_r2_step > best_r2_overall:
            selected.append(best_feat_step)
            remaining.remove(best_feat_step)
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
        import json
        p = OUTPUT_DIR / f'info_{name}.json'
        with open(p, 'w') as f:
            json.dump({'features': features}, f, indent=2)
        # mlflow.log_artifact(str(p))




# ============================================================================
# Experiments
# ============================================================================
def exp_F_remove_age(X, y, data_hash):
    """F: Forward stepwise selection entirely without 'current_age'."""
    print(f"\n{'='*60}\nEXP F: Remove Age\n{'='*60}")
    
    # Restrict feature pool
    X_pool = X.drop(columns=['current_age'], errors='ignore')
    
    # Run stepwise Ridge
    feats, y_pred, metrics = forward_stepwise(X_pool, y.values, max_features=15, alpha=10.0)
    name = 'F1_Stepwise_No_Age'
    log_run(name, 'restriction', {'restriction': 'no_age'}, metrics, feats, data_hash)
    
    print(f"  F1) No Age (Stepwise, {len(feats)} features): R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")
    print(f"      Features: {feats[:5]}...")
    return {
        name: {
            'metrics': metrics,
            'y_pred': y_pred,
            'features': feats,
            'model_name': 'Ridge_NoAge'
        }
    }


def exp_G_bilateral_mean(X, y, data_hash):
    """G: Stepwise selection restricted to _mean, _ratio, and systemic vessels. No left/right exclusive features."""
    print(f"\n{'='*60}\nEXP G: Enforce Bilateral Mean\n{'='*60}")
    
    # Define valid features
    valid_cols = []
    for c in X.columns:
        c_low = c.lower()
        if 'age' in c_low:
            valid_cols.append(c)
        elif 'mean' in c_low or 'ratio' in c_low:
            valid_cols.append(c)
        elif 'aorta' in c_low or 'venacava' in c_low or 'hepatic' in c_low or 'portal' in c_low:
            # Systemic vessels are allowed
            valid_cols.append(c)
            
    X_pool = X[valid_cols]
    
    # Run stepwise Ridge
    feats, y_pred, metrics = forward_stepwise(X_pool, y.values, max_features=15, alpha=10.0)
    name = 'G1_Bilateral_Mean_Only'
    log_run(name, 'restriction', {'restriction': 'bilateral_mean_only'}, metrics, feats, data_hash)
    
    print(f"  G1) Bilateral Mean (Stepwise, {len(feats)} features): R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")
    print(f"      Features: {feats[:5]}...")
    return {
        name: {
            'metrics': metrics,
            'y_pred': y_pred,
            'features': feats,
            'model_name': 'Ridge_BilateralMean'
        }
    }


def exp_H_strict_kidney(X, y, data_hash):
    """H: Stepwise selection without age, without non-kidney/non-aorta vessels."""
    print(f"\n{'='*60}\nEXP H: Strict Kidney-Only Model\n{'='*60}")
    
    # Define valid features
    valid_cols = []
    for c in X.columns:
        c_low = c.lower()
        # Only kidney measurements and aorta (input function)
        if 'kidney' in c_low or 'aorta' in c_low or 'mean_' in c_low or '^e_' in c_low or 'lr_ratio' in c_low or 'phase_contrast' in c_low:
            if 'venacava' not in c_low and 'hepatic' not in c_low and 'portal' not in c_low and 'age' not in c_low:
                valid_cols.append(c)
                
    # Double check E_ phases
    valid_cols = [c for c in X.columns if c in valid_cols or c.startswith('E_')]
    
    X_pool = X[valid_cols]
    
    # Run stepwise Ridge
    feats, y_pred, metrics = forward_stepwise(X_pool, y.values, max_features=15, alpha=10.0)
    name = 'H1_Strict_Kidney_Only'
    log_run(name, 'restriction', {'restriction': 'strict_kidney'}, metrics, feats, data_hash)
    
    print(f"  H1) Strict Kidney Only (Stepwise, {len(feats)} features): R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")
    print(f"      Features: {feats[:5]}...")
    
    # Run BayesianRidge with these features
    est = Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())])
    y_pred_br, metrics_br = evaluate_loocv(X_pool[feats].values, y.values, est)
    name_br = 'H2_Strict_Kidney_Bayesian'
    log_run(name_br, 'restriction', {'restriction': 'strict_kidney', 'model': 'BayesianRidge'}, metrics_br, feats, data_hash)
    print(f"  H2) Strict Kidney BayesianRidge: R2={metrics_br['R2']:.3f}  MAE={metrics_br['MAE']:.2f}")
    return {
        name: {
            'metrics': metrics,
            'y_pred': y_pred,
            'features': feats,
            'model_name': 'Ridge_StrictKidney'
        },
        name_br: {
            'metrics': metrics_br,
            'y_pred': y_pred_br,
            'features': feats,
            'model_name': 'BayesianRidge_StrictKidney'
        }
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR Round 3 Improvements (Feature Restrictions)")
    print("  R2 best: BayesianRidge Stepwise R2=0.574, MAE=7.53")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, data_hash = load_and_prepare()
    print(f"[OK] {len(X)} patients, {len(X.columns)} features total")

    all_results = {}

    with mlflow.start_run(run_name=f"round3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'round3_improvements')
        mlflow.log_param('data_hash', data_hash)

        # F: Remove Age
        for k, v in exp_F_remove_age(X, y, data_hash).items():
            all_results[k] = v

        # G: Bilateral Mean
        for k, v in exp_G_bilateral_mean(X, y, data_hash).items():
            all_results[k] = v

        # H: Strict Kidney Only
        for k, v in exp_H_strict_kidney(X, y, data_hash).items():
            all_results[k] = v

        # Unpack metrics for the dataframe
        results_metrics = {k: v['metrics'] for k, v in all_results.items()}
        results_df = pd.DataFrame(results_metrics).T.sort_values('R2', ascending=False)
        results_df.to_csv(OUTPUT_DIR / 'round3_results.csv')
        mlflow.log_artifact(str(OUTPUT_DIR / 'round3_results.csv'))

        champion_name = results_df.index[0]
        champ_data = all_results[champion_name]
        champ_metrics = champ_data['metrics']
        champ_y_pred = champ_data['y_pred']
        champ_features = champ_data['features']
        model_label = f"R3 Champ (Restricted) - {champ_data['model_name']}"

        # 1. Main Scatter Plot
        plot_path = OUTPUT_DIR / f'round3_champion_{champ_data["model_name"]}.png'
        plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, champ_features, champ_metrics, plot_path)
        mlflow.log_artifact(str(plot_path))

        # 2. Residual Plot
        res_path = OUTPUT_DIR / f'round3_champion_{champ_data["model_name"]}_residuals.png'
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

        print(f"\n{'='*60}")
        print("ROUND 3 RESULTS")
        print(f"{'='*60}")
        for name, row in results_df.iterrows():
            marker = " <-- BEST" if name == results_df.index[0] else ""
            print(f"  {name:30s}  R2={row['R2']:+.3f}  MAE={row['MAE']:.2f}{marker}")

if __name__ == '__main__':
    main()
