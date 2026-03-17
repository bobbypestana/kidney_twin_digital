"""
vGFR Model Improvement Experiments -- Round 6 (MAE & RMSE Competition)

Experiment L: 
  The user correctly pointed out that taking the features optimized for R2 
  and evaluating them on MAE/RMSE isn't a fair comparison to models optimized 
  specifically for those metrics.
  
  This script evaluates a suite of diverse algorithms (Ridge, Lasso, ElasticNet, 
  HuberRegressor, RandomForest, SVR, GP), using forward stepwise selection 
  where the internal loss function is STRICTLY MAE or RMSE.
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

# ============================================================================
# Config
# ============================================================================
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
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo, n_jobs=-1)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }


def stepwise_optimize_metric(X, y_arr, base_estimator, metric='MAE', max_features=10):
    """
    Forward stepwise selection that directly minimizes the target metric.
    """
    selected = []
    y_pred_best = None
    best_metrics = None
    best_score_overall = np.inf

    remaining = list(X.columns)

    while len(selected) < max_features and remaining:
        best_score_step = np.inf
        best_feat_step = None
        best_pred_step = None
        best_metrics_step = None

        for feat in remaining:
            trial_feats = selected + [feat]
            
            est = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            
            score = metrics[metric]
            if score < best_score_step:
                best_score_step = score
                best_feat_step = feat
                best_pred_step = y_pred
                best_metrics_step = metrics

        if best_score_step < best_score_overall:
            selected.append(best_feat_step)
            remaining.remove(best_feat_step)
            best_score_overall = best_score_step
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
def eval_all_models(X, y, data_hash, target_metric):
    print(f"\n{'='*60}\nEvaluating all models for: {target_metric}\n{'='*60}")
    
    # Pre-filter top 20 features by correlation to speed up cross-validation across 7 models
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    pool = corr.head(20).index.tolist()
    if 'current_age' not in pool and 'current_age' in X.columns:
        pool.append('current_age')
    X_pool = X[pool]
    
    models = {
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Huber': HuberRegressor(epsilon=1.35),
        'BayesianRidge': BayesianRidge(),
        'RandomForest': RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=10.0, gamma='scale')
    }
    
    results = {}
    
    for model_name, estimator in models.items():
        print(f"Training {model_name}...")
        feats, y_pred, metrics = stepwise_optimize_metric(
            X_pool, y.values, estimator, metric=target_metric, max_features=10
        )
        
        name = f"L_{model_name}_{target_metric}_Optimized"
        
        # Ensure we don't crash if model selected 0 features
        if len(feats) > 0:
            log_run(name, f'{target_metric}_competition', {'model': model_name}, metrics, feats, data_hash)
            results[name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'features': feats,
                'model_name': model_name
            }
            
            print(f"  --> {model_name}: {target_metric}={metrics[target_metric]:.2f} (R2={metrics['R2']:.3f}) with {len(feats)} features")
        else:
            print(f"  --> {model_name}: Failed to select any features that improved {target_metric}")
            
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR Round 6 Improvements (MAE & RMSE Competition)")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, data_hash = load_and_prepare()
    print(f"[OK] {len(X)} patients, {len(X.columns)} features total")

    with mlflow.start_run(run_name=f"round6_competition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'round6_competition')
        mlflow.log_param('data_hash', data_hash)

        # 1. Optimize everyone for MAE
        mae_results = eval_all_models(X, y, data_hash, 'MAE')
        
        # 2. Optimize everyone for RMSE
        rmse_results = eval_all_models(X, y, data_hash, 'RMSE')

        # Save MAE leaderboard
        mae_metrics = {k: v['metrics'] for k, v in mae_results.items()}
        mae_df = pd.DataFrame(mae_metrics).T.replace([np.inf, -np.inf], np.nan).dropna(subset=['MAE']).sort_values('MAE', ascending=True)
        # mae_df.to_csv(OUTPUT_DIR / 'round6_mae_leaderboard.csv')
        
        # DEFINITION RESTORED:
        rmse_metrics = {k: v['metrics'] for k, v in rmse_results.items()}
        rmse_df = pd.DataFrame(rmse_metrics).T.replace([np.inf, -np.inf], np.nan).dropna(subset=['RMSE']).sort_values('RMSE', ascending=True)
        # rmse_df.to_csv(OUTPUT_DIR / 'round6_rmse_leaderboard.csv')

        print(f"\n{'='*60}")
        print("MAE RANKINGS")
        for name, row in mae_df.iterrows():
            print(f"  {name:35s}  MAE={row['MAE']:.2f}")

        print(f"\n{'='*60}")
        print("RMSE RANKINGS")
        for name, row in rmse_df.iterrows():
            print(f"  {name:35s}  RMSE={row['RMSE']:.2f}")

        # --- Plot Champions ---
        def plot_champion(champ_key, all_res_dict, title_prefix, file_prefix):
            champ_data = all_res_dict[champ_key]
            champ_metrics = champ_data['metrics']
            champ_y_pred = champ_data['y_pred']
            champ_features = champ_data['features']
            model_label = f"{title_prefix} - {champ_data['model_name']}"

            # Main
            plot_path = OUTPUT_DIR / f'{file_prefix}_{champ_data["model_name"]}.png'
            plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, champ_features, champ_metrics, plot_path)
            mlflow.log_artifact(str(plot_path))

            # Residuals
            res_path = OUTPUT_DIR / f'{file_prefix}_{champ_data["model_name"]}_residuals.png'
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
            print(f"[OK] Saved Champion {title_prefix} Residuals: {res_path}")

        if not mae_df.empty:
            plot_champion(mae_df.index[0], mae_results, "R6 MAE Champ", "round6_champion_mae")
        if not rmse_df.empty:
            plot_champion(rmse_df.index[0], rmse_results, "R6 RMSE Champ", "round6_champion_rmse")

if __name__ == '__main__':
    main()
