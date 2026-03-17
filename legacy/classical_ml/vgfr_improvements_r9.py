"""
vGFR Model Improvement Experiments -- Round 9 (Minimum N Feature Forcing WITHOUT Age)

Experiment N: 
  Exactly like Round 8 (forcing greedy selection past local optimums to reach N=7),
  but we strictly REMOVE `current_age` from the dataset.
  
  The goal is to see if, without the dominant age feature, forcing the algorithm 
  to combine more CT features allows it to discover deeper structural patterns.
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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import mlflow
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path('database/egfr_data.duckdb')
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data.duckdb'

OUTPUT_DIR = Path('classical_ml/ml_results')
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path(__file__).parent / 'ml_results'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR = Path('C:/tmp/mlflow_vgfr')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "vGFR_Kidney_Digital_Twin"
TARGET = 'egfrc'

EXCLUDE_PREFIXES = ['vgfr_', 'conc_lit_', 'conc_late_', 'w_pv_', 'w_back_']
EXCLUDE_COLS = [
    'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
    'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days', TARGET,
    'current_age' # <-- KEY CHANGE: Excluding age
]

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

    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median()), y, data_hash

def evaluate_loocv(X_arr, y_arr, estimator):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo, n_jobs=-1)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred)
    }

def forward_stepwise_forced_N(X, y_arr, base_estimator, metric='MAE', force_N=7, max_features=12):
    selected = []
    y_pred_best = None
    best_metrics = None
    true_best_score = np.inf 
    
    remaining = list(X.columns)
    
    # Pre-select to save time
    corr = X.corrwith(pd.Series(y_arr, index=X.index) if isinstance(y_arr, np.ndarray) else y_arr).abs().sort_values(ascending=False)
    pool = corr.head(25).index.tolist()
    remaining = [f for f in remaining if f in pool]

    for step in range(1, max_features + 1):
        if not remaining: break
        
        best_score_this_step = np.inf
        best_feat_this_step = None
        best_pred_this_step = None
        best_metrics_this_step = None

        for feat in remaining:
            trial_feats = selected + [feat]
            est = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            
            score = metrics[metric] 
            
            if score < best_score_this_step:
                best_score_this_step = score
                best_feat_this_step = feat
                best_pred_this_step = y_pred
                best_metrics_this_step = metrics

        is_improvement = best_score_this_step < true_best_score
        
        if is_improvement or len(selected) < force_N:
            selected.append(best_feat_this_step)
            remaining.remove(best_feat_this_step)
            
            if is_improvement:
                true_best_score = best_score_this_step
                y_pred_best = best_pred_this_step
                best_metrics = best_metrics_this_step
        else:
            break

    # Replay to find exact point of optimal stopping
    final_selected = []
    for i in range(1, len(selected) + 1):
        trial = selected[:i]
        est = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
        yp, m = evaluate_loocv(X[trial].values, y_arr, est)
        if m[metric] <= best_metrics[metric] + 1e-5:
            final_selected = trial
            break
            
    return final_selected, y_pred_best, best_metrics

def main():
    print("=" * 60)
    print("  vGFR Round 9 Improvements (Forced N WITHOUT Age)")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, data_hash = load_and_prepare()
    print(f"[OK] {len(X)} patients, {len(X.columns)} features total")

    models = {
        'SVR': SVR(kernel='rbf', C=10.0, gamma='scale'),
        'Ridge': Ridge(alpha=10.0),
        'BayesianRidge': BayesianRidge()
    }
    
    all_results = {}
    FORCE_N = 7

    with mlflow.start_run(run_name=f"round9_noAge_forcedN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'round9_noAge_forcedN')
        mlflow.log_param('data_hash', data_hash)

        for metric in ['MAE', 'RMSE']:
            print(f"\n--- Optimizing for {metric} (No Age, Forced Min N={FORCE_N}) ---")
            for name, estimator in models.items():
                print(f"Forcing {name} through local optima...")
                feats, y_pred, metrics = forward_stepwise_forced_N(X, y.values, estimator, metric=metric, force_N=FORCE_N, max_features=12)
                
                exp_name = f"N1_{name}_{metric}_NoAge"
                
                # Log
                with mlflow.start_run(run_name=exp_name, nested=True):
                    mlflow.set_tag('experiment_type', 'forced_N_no_age')
                    mlflow.log_params({'model': name, 'metric': metric, 'force_N': FORCE_N})
                    mlflow.log_metrics(metrics)
                
                all_results[exp_name] = {
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'features': feats,
                    'model_name': name,
                    'metric_opt': metric
                }
                
                print(f"  --> MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f}")
                print(f"  --> Final Features Retained: {len(feats)}")

        # results_df.to_csv(OUTPUT_DIR / 'round9_results.csv')
        
        results_df = pd.DataFrame({k: v['metrics'] for k, v in all_results.items()}).T

        def calc_blend(row):
            return (row['MAE']/10.0) + (row['RMSE']/15.0) - row['R2']
            
        results_df['CompositeScore'] = results_df.apply(calc_blend, axis=1)
        results_df = results_df.sort_values('CompositeScore', ascending=True)
        
        for name, row in results_df.iterrows():
            print(f"  {name:35s}  Composite={row['CompositeScore']:.3f} (MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, R2={row['R2']:.3f})")

        champion_name = results_df.index[0]
        champ_data = all_results[champion_name]
        champ_metrics = champ_data['metrics']
        champ_y_pred = champ_data['y_pred']
        champ_features = champ_data['features']
        model_label = f"R9 Champ - {champ_data['model_name']} (Opt:{champ_data['metric_opt']}, NoAge)"

        # 1. Main Scatter Plot
        plot_path = OUTPUT_DIR / f'round9_champion_{champ_data["model_name"]}.png'
        plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, champ_features, champ_metrics, plot_path)
        mlflow.log_artifact(str(plot_path))

        # 2. Residual Plot
        res_path = OUTPUT_DIR / f'round9_champion_{champ_data["model_name"]}_residuals.png'
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
