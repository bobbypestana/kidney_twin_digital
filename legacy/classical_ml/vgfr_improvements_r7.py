"""
vGFR Model Improvement Experiments -- Round 7 (Composite Score Feature Selection)

Experiment L: 
  Instead of optimizing for just one metric, this stepwise selection blends 
  R2, MAE, and RMSE.
  
  For each step, we evaluate all remaining candidate features. We rank them 
  based on MAE (lowest is rank 1), RMSE (lowest is rank 1), and R2 (highest is rank 1).
  The feature with the best (lowest) average rank across all three metrics is selected.
  This forces the algorithm to pick "balanced" features.
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


def forward_stepwise_blended_rank(X, y_arr, base_estimator, max_features=12):
    """
    Greedy forward selection that evaluates features using a composite rank 
    across MAE, RMSE, and R2.
    """
    selected = []
    y_pred_best = None
    best_metrics = None
    
    # Baseline model without any features just predicts mean
    best_composite_score = np.inf 

    remaining = list(X.columns)
    
    # Pre-select top candidates to reduce search space for expensive models like RF/SVR
    # Note: need to calculate correlation before passing numpy arrays
    corr = X.corrwith(pd.Series(y_arr, index=X.index) if isinstance(y_arr, np.ndarray) else y_arr).abs().sort_values(ascending=False)
    pool = corr.head(25).index.tolist()
    if 'current_age' not in pool and 'current_age' in X.columns:
        pool.append('current_age')
    remaining = [f for f in remaining if f in pool]

    while len(selected) < max_features and remaining:
        # Collect results for all candidate features at this step
        step_results = []
        
        for feat in remaining:
            trial_feats = selected + [feat]
            est = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
            y_pred, metrics = evaluate_loocv(X[trial_feats].values, y_arr, est)
            
            step_results.append({
                'feature': feat,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'metrics_dict': metrics,
                'y_pred': y_pred
            })

        # Rank all candidates
        df_step = pd.DataFrame(step_results)
        df_step['Rank_MAE'] = df_step['MAE'].rank(ascending=True) # Lowest MAE is 1st
        df_step['Rank_RMSE'] = df_step['RMSE'].rank(ascending=True) # Lowest RMSE is 1st
        df_step['Rank_R2'] = df_step['R2'].rank(ascending=False) # Highest R2 is 1st
        df_step['Avg_Rank'] = df_step[['Rank_MAE', 'Rank_RMSE', 'Rank_R2']].mean(axis=1)
        
        # Sort to find the absolute best blended candidate
        df_best = df_step.sort_values('Avg_Rank').iloc[0]
        
        best_feat_step = df_best['feature']
        best_metrics_step = df_best['metrics_dict']
        best_pred_step = df_best['y_pred']
        
        # We need to decide if we should stop. We can stop if the composite loss doesn't improve.
        # Let's define a composite loss logic here just for early stopping: 
        # Loss = (MAE/10.0) + (RMSE/15.0) - R2
        composite_loss_step = (best_metrics_step['MAE']/10.0) + (best_metrics_step['RMSE']/15.0) - best_metrics_step['R2']
        
        if composite_loss_step < best_composite_score:
            selected.append(best_feat_step)
            remaining.remove(best_feat_step)
            best_composite_score = composite_loss_step
            y_pred_best = best_pred_step
            best_metrics = best_metrics_step
        else:
            # Adding more features doesn't improve our overall composite loss, stop here
            break

    return selected, y_pred_best, best_metrics


def log_run(name, tag, params, metrics, features, data_hash):
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tag('experiment_type', tag)
        mlflow.log_params({**params, 'data_hash': data_hash})
        mlflow.log_metrics(metrics)
        # import json
        # p = OUTPUT_DIR / f'info_{name}.json'
        # with open(p, 'w') as f:
        #     json.dump({'features': features}, f, indent=2)
        # mlflow.log_artifact(str(p))



# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR Round 7 Improvements (Composite Stepwise Selection)")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, data_hash = load_and_prepare()
    print(f"[OK] {len(X)} patients, {len(X.columns)} features total")

    # Let's run this for both Ridge and BayesianRidge, as they are our top performers
    models = {
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Huber': HuberRegressor(epsilon=1.35),
        'BayesianRidge': BayesianRidge(),
        'RandomForest': RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=10.0, gamma='scale')
    }
    
    all_results = {}

    with mlflow.start_run(run_name=f"round7_blended_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'round7_composite_search')
        mlflow.log_param('data_hash', data_hash)

        for name, estimator in models.items():
            print(f"Running Blended Rank Stepwise for {name}...")
            feats, y_pred, metrics = forward_stepwise_blended_rank(X, y.values, estimator, max_features=12)
            
            exp_name = f"L1_Blended_{name}"
            log_run(exp_name, 'composite_optimization', {'model': name}, metrics, feats, data_hash)
            all_results[exp_name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'features': feats,
                'model_name': name
            }
            
            print(f"  --> MAE: {metrics['MAE']:.2f}")
            print(f"  --> RMSE: {metrics['RMSE']:.2f}")
            print(f"  --> R2: {metrics['R2']:.3f}")
            print(f"  --> Features Selected: {len(feats)} {feats}\n")
            
        results_df = pd.DataFrame({k: v['metrics'] for k, v in all_results.items()}).T
        # results_df.to_csv(OUTPUT_DIR / 'round7_results.csv')
        # mlflow.log_artifact(str(OUTPUT_DIR / 'round7_results.csv'))

        print(f"\n{'='*60}")
        print("BLENDED RANK LEADERBOARD (Sorted by custom composite score)")
        print(f"{'='*60}")
        # Sort by the score we used for early stopping to find the best all-rounder
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
        model_label = f"R7 Champ - {champ_data['model_name']} (Composite)"

        # 1. Main Scatter Plot
        plot_path = OUTPUT_DIR / f'round7_champion_{champ_data["model_name"]}.png'
        plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, champ_features, champ_metrics, plot_path)
        mlflow.log_artifact(str(plot_path))

        # 2. Residual Plot
        res_path = OUTPUT_DIR / f'round7_champion_{champ_data["model_name"]}_residuals.png'
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
