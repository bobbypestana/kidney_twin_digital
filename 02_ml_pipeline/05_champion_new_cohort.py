"""
Consolidated Champion Model for New Cohort (12-03-2026)
Architecture: Stacking Ensemble (Round 12)
Features: Demographic-aware (Age+Sex) + Best Vascular Markers.
Excludes: Volume, HU, Chemistry (No serum_creatinine).
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, HuberRegressor, BayesianRidge
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Config
DB_PATH = Path('g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb')
OUTPUT_DIR = Path('g:/My Drive/kvantify/DanQ_health/analysis/02_ml_pipeline/ml_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'egfrc'
COHORT = '12-03-2026'

def load_data():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute(f"SELECT * FROM gold.ml_features WHERE source_folder = '{COHORT}'").df()
    conn.close()
    
    # Map Sex to numeric
    df['sex_num'] = df['sex'].map({'M': 1, 'F': 0}).fillna(0.5)
    
    # Strictly exclude non-image features
    meta_cols = [
        'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
        'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days',
        'source_folder', TARGET
    ]
    
    X = df.drop(columns=[c for c in meta_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    
    # Exclude Volume, HU, Statistics and Chemistry
    excluded_patterns = ['vol', 'hu', 'std', 'median', 'creatinine']
    X = X.drop(columns=[c for c in X.columns if any(p in c.lower() for p in excluded_patterns)])
    
    X = X.astype(float).fillna(X.median()).fillna(0)
    y = df[TARGET]
    return X, y, df['sex']

def evaluate_loocv(X_arr, y_arr, estimator):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo, n_jobs=-1)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }

def stepwise_selection(X, y_arr, base_estimator, forced=None, max_total=10):
    selected = forced if forced else []
    remaining = [c for c in X.columns if c not in selected]
    best_r2 = -np.inf
    
    # Initial R2 with forced features
    if selected:
        pipe = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
        _, m = evaluate_loocv(X[selected].values, y_arr, pipe)
        best_r2 = m['R2']
        print(f"  Starting with: {', '.join(selected)} (R2={best_r2:.4f})")

    y_pred_best = None
    best_metrics = None

    while len(selected) < max_total and remaining:
        best_r2_step = -np.inf
        best_feat_step = None
        for feat in remaining:
            trial = selected + [feat]
            pipe = Pipeline([('scaler', StandardScaler()), ('model', base_estimator)])
            y_pred, metrics = evaluate_loocv(X[trial].values, y_arr, pipe)
            if metrics['R2'] > best_r2_step:
                best_r2_step = metrics['R2']
                best_feat_step = feat
                best_pred_step = y_pred
                best_metrics_step = metrics
        
        if best_r2_step > best_r2:
            selected.append(best_feat_step)
            remaining.remove(best_feat_step)
            best_r2 = best_r2_step
            y_pred_best = best_pred_step
            best_metrics = best_metrics_step
            print(f"  Added '{best_feat_step}' (R2={best_r2:.4f})")
        else:
            break
    return selected, y_pred_best, best_metrics

def main():
    X, y, sex = load_data()
    print(f"Training Demographic-Aware Champion on {COHORT}")
    
    # 1. Feature Selection (Forcing Age and Sex)
    forced = ['current_age', 'sex_num']
    selected_features, _, _ = stepwise_selection(X, y.values, HuberRegressor(), forced=forced, max_total=10)
    
    # 2. Train Stacking Ensemble on selected features
    print(f"Evaluating Stacking Ensemble on {len(selected_features)} features...")
    estimators = [
        ('bayesian', BayesianRidge()),
        ('huber', HuberRegressor()),
        ('svr', SVR(kernel='rbf', C=10.0))
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
    
    X_selected = X[selected_features].values
    pipe = Pipeline([('scaler', StandardScaler()), ('model', stack)])
    y_pred, metrics = evaluate_loocv(X_selected, y.values, pipe)
    
    print("\n" + "="*40)
    print("DEMOGRAPHIC-AWARE CHAMPION RESULTS")
    print("="*40)
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2:   {metrics['R2']:.3f}")
    print(f"Features: {', '.join(selected_features)}")
    
    # Plot
    plot_path = OUTPUT_DIR / f"{COHORT.replace('-', '')}_stacking_champion.png"
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_pred, y=y, hue=sex, s=100)
    lims = [0, 120]
    plt.plot(lims, lims, '--r', alpha=0.5)
    plt.title(f"Demographic-Aware Champion | Stacking\nMAE={metrics['MAE']:.2f}, R2={metrics['R2']:.3f}")
    plt.xlabel("Predicted vGFR")
    plt.ylabel("Actual eGFRc")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path, dpi=300)
    print(f"\nChampion plot saved: {plot_path.name}")

if __name__ == "__main__":
    main()
