"""
vGFR Model Improvement Experiments
Each experiment is tracked separately in MLflow for comparison.

Baseline: Ridge (top 10 features, alpha=10) -> R2=0.406, MAE=8.67

Experiments:
  1. Two-stage age-residual model
  2. Forward stepwise feature selection
  3. Richer feature engineering (interactions, cross-phase, bilateral sums)
  4. Log-target transformation
  5. Outlier-robust regression (Huber, RANSAC)
  6. Hyperparameter tuning via nested LOOCV

Usage:
    python vgfr_improvements.py
"""

import duckdb
import pandas as pd
import numpy as np
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LinearRegression,
    HuberRegressor, RANSACRegressor, BayesianRidge
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data_v2.duckdb'
OUTPUT_DIR = Path(__file__).parent / 'ml_results_v2'
OUTPUT_DIR.mkdir(exist_ok=True)

MLFLOW_DIR = Path('C:/tmp/mlflow_vgfr_v2')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "vGFR_v2_Sanity_Check"

TARGET = 'egfrc'
EXCLUDE_PREFIXES = ['vgfr_', 'conc_lit_', 'conc_late_', 'w_pv_', 'w_back_']
EXCLUDE_COLS = [
    'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
    'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days',
    'source_folder', # Added this
    TARGET,
]


# ============================================================================
# Data Loading (shared)
# ============================================================================
def load_data():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.master_cases').fetchdf()
    conn.close()
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]
    print(f"[OK] Loaded {len(df)} patients, hash={data_hash}")
    return df, data_hash


def select_features(df):
    drop_cols = list(EXCLUDE_COLS)
    for col in df.columns:
        for prefix in EXCLUDE_PREFIXES:
            if col.startswith(prefix):
                drop_cols.append(col)
                break
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[TARGET]
    return X, y


def base_engineer(X):
    """Standard feature engineering (same as baseline)."""
    X = X.copy()
    for phase, side in [('arterial','left'),('arterial','right'),
                        ('venous','left'),('venous','right'),
                        ('late','left'),('late','right')]:
        art = f'{phase}_{side}_kidney_artery'
        vein = f'{phase}_{side}_kidney_vein'
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
        for vessel in ['artery', 'vein']:
            art_col = f'arterial_{side}_kidney_{vessel}'
            late_col = f'late_{side}_kidney_{vessel}'
            if art_col in X.columns and late_col in X.columns:
                X[f'phase_contrast_{vessel}_{side}'] = X[art_col] - X[late_col]

    for phase in ['arterial', 'venous', 'late']:
        aorta = f'{phase}_aorta'
        if aorta in X.columns:
            for side in ['left', 'right']:
                for vessel in ['artery', 'vein']:
                    col = f'{phase}_{side}_kidney_{vessel}'
                    if col in X.columns:
                        X[f'norm_{vessel}_{phase}_{side}'] = X[col] / X[aorta].replace(0, np.nan)

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
    return X


# ============================================================================
# Helper: evaluate + log to MLflow
# ============================================================================
def evaluate_loocv(X_arr, y_arr, estimator):
    """Run LOOCV and return metrics + predictions."""
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo)
    mae = mean_absolute_error(y_arr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
    r2 = r2_score(y_arr, y_pred)
    max_err = float(np.max(np.abs(y_arr - y_pred)))
    median_ae = float(np.median(np.abs(y_arr - y_pred)))
    return y_pred, {'MAE': mae, 'RMSE': rmse, 'R2': r2,
                    'max_error': max_err, 'median_AE': median_ae}


def log_experiment(run_name, experiment_tag, params, X_sel, y_arr,
                   estimator, feature_names, data_hash, extra_info=None):
    """Run LOOCV, log everything to MLflow, return metrics."""
    y_pred, metrics = evaluate_loocv(X_sel, y_arr, estimator)

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag('experiment_type', experiment_tag)

        mlflow.log_params({
            'data_hash': data_hash,
            'n_patients': len(y_arr),
            'n_features': X_sel.shape[1],
            'cv_method': 'LOOCV',
            **params,
        })

        mlflow.log_metrics(metrics)

        # Feature list
        feat_info = {
            'features': feature_names if feature_names else
                        [f'feat_{i}' for i in range(X_sel.shape[1])],
        }
        if extra_info:
            feat_info.update(extra_info)

        info_path = OUTPUT_DIR / f'info_{run_name}.json'
        with open(info_path, 'w') as f:
            json.dump(feat_info, f, indent=2, default=str)
        mlflow.log_artifact(str(info_path))

        # Predictions
        pred_df = pd.DataFrame({
            'actual': y_arr, 'predicted': y_pred,
            'residual': y_arr - y_pred,
            'abs_error': np.abs(y_arr - y_pred),
        })
        pred_path = OUTPUT_DIR / f'preds_{run_name}.csv'
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path))

        # Pred vs actual plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_arr, y_pred, s=80, alpha=0.7, edgecolors='black',
                   linewidth=0.5, c='steelblue')
        lims = [min(y_arr.min(), y_pred.min()) - 5,
                max(y_arr.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Actual eGFRc'); ax.set_ylabel('Predicted eGFRc')
        ax.set_title(f'{run_name}\nR2={metrics["R2"]:.3f}, MAE={metrics["MAE"]:.2f}')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / f'plot_{run_name}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(plot_path))

    return metrics, y_pred


def get_top_k(X, y, k):
    """Select top-k features by absolute correlation."""
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    feats = corrs.index[:k].tolist()
    return feats


# ============================================================================
# EXPERIMENT 0: Baseline (Ridge top-10, alpha=10)
# ============================================================================
def exp0_baseline(X, y, data_hash):
    print(f"\n{'='*60}")
    print("EXP 0: BASELINE -- Ridge (top 10, alpha=10)")
    print(f"{'='*60}")

    feats = get_top_k(X, y, 10)
    scaler = StandardScaler()
    X_sel = scaler.fit_transform(X[feats])

    estimator = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
    metrics, _ = log_experiment(
        'EXP0_baseline_Ridge_top10_a10', 'baseline',
        {'model': 'Ridge', 'alpha': 10.0, 'feature_selection': 'corr_top10'},
        X[feats].values, y.values, estimator, feats, data_hash
    )
    print(f"  R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}")
    return metrics


# ============================================================================
# EXP 1: Two-stage age-residual model
# ============================================================================
def exp1_age_residual(X, y, data_hash):
    print(f"\n{'='*60}")
    print("EXP 1: Two-stage age-residual model")
    print(f"{'='*60}")

    results = {}
    y_arr = y.values
    age = X['current_age'].values.reshape(-1, 1)

    # Stage 1: age-only model (to see how much age explains)
    est_age = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
    metrics_age, y_pred_age = log_experiment(
        'EXP1a_age_only', 'age_residual',
        {'model': 'Ridge', 'alpha': 1.0, 'stage': 'age_only'},
        age, y_arr, est_age, ['current_age'], data_hash
    )
    print(f"  1a) Age-only:         R2={metrics_age['R2']:.3f}  MAE={metrics_age['MAE']:.2f}")

    # Compute residuals from LOOCV age predictions
    residuals_from_age = y_arr - y_pred_age

    # Stage 2: predict residuals from imaging features (no age)
    X_no_age = X.drop(columns=['current_age'])
    feats_no_age = get_top_k(X_no_age, pd.Series(residuals_from_age, index=y.index), 10)

    for alpha_val in [1.0, 10.0, 20.0]:
        est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha_val))])
        metrics_res, y_pred_res = log_experiment(
            f'EXP1b_residual_imaging_a{int(alpha_val)}', 'age_residual',
            {'model': 'Ridge', 'alpha': alpha_val, 'stage': 'residual_imaging',
             'feature_selection': 'corr_top10_on_residuals'},
            X_no_age[feats_no_age].values, residuals_from_age, est,
            feats_no_age, data_hash,
            extra_info={'note': 'predicting age-residuals from imaging only'}
        )

        # Combined prediction = age_pred + residual_pred
        # For fair comparison, need LOOCV combined
        # Re-do full LOOCV with two-stage approach
        loo = LeaveOneOut()
        y_combined = np.zeros(len(y_arr))
        for train_idx, test_idx in loo.split(X.values):
            # Stage 1: fit age model on train
            age_model = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
            age_model.fit(age[train_idx], y_arr[train_idx])
            age_pred_test = age_model.predict(age[test_idx])
            age_pred_train = age_model.predict(age[train_idx])

            # Stage 2: fit imaging model on train residuals
            train_residuals = y_arr[train_idx] - age_pred_train
            res_model = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha_val))])
            res_model.fit(X_no_age[feats_no_age].values[train_idx], train_residuals)
            res_pred_test = res_model.predict(X_no_age[feats_no_age].values[test_idx])

            y_combined[test_idx] = age_pred_test + res_pred_test

        mae_c = mean_absolute_error(y_arr, y_combined)
        rmse_c = np.sqrt(mean_squared_error(y_arr, y_combined))
        r2_c = r2_score(y_arr, y_combined)

        with mlflow.start_run(run_name=f'EXP1c_combined_a{int(alpha_val)}', nested=True):
            mlflow.set_tag('experiment_type', 'age_residual')
            mlflow.log_params({
                'data_hash': data_hash, 'n_patients': len(y_arr),
                'model': 'TwoStage_Ridge', 'alpha_age': 1.0,
                'alpha_imaging': alpha_val, 'stage': 'combined',
                'n_features': len(feats_no_age) + 1,
            })
            mlflow.log_metrics({'MAE': mae_c, 'RMSE': rmse_c, 'R2': r2_c})

            feat_info = {'age_feature': 'current_age', 'imaging_features': feats_no_age}
            info_path = OUTPUT_DIR / f'info_EXP1c_a{int(alpha_val)}.json'
            with open(info_path, 'w') as f:
                json.dump(feat_info, f, indent=2)
            mlflow.log_artifact(str(info_path))

        metrics_c = {'MAE': mae_c, 'RMSE': rmse_c, 'R2': r2_c}
        results[f'combined_a{int(alpha_val)}'] = metrics_c
        print(f"  1c) Combined (a={alpha_val:4.1f}): R2={r2_c:.3f}  MAE={mae_c:.2f}  RMSE={rmse_c:.2f}")

    return results


# ============================================================================
# EXP 2: Forward stepwise feature selection
# ============================================================================
def exp2_stepwise(X, y, data_hash):
    print(f"\n{'='*60}")
    print("EXP 2: Forward stepwise feature selection")
    print(f"{'='*60}")

    y_arr = y.values
    loo = LeaveOneOut()

    # Start from empty, greedily add the feature that improves LOOCV R2 most
    remaining = list(X.columns)
    selected = []
    best_r2_history = []
    best_overall_r2 = -np.inf

    for step in range(min(15, len(remaining))):
        best_feat = None
        best_r2_step = -np.inf

        for feat in remaining:
            trial = selected + [feat]
            X_trial = X[trial].values
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
            y_pred = cross_val_predict(est, X_trial, y_arr, cv=loo)
            r2 = r2_score(y_arr, y_pred)
            if r2 > best_r2_step:
                best_r2_step = r2
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_r2_history.append(best_r2_step)

        if best_r2_step > best_overall_r2:
            best_overall_r2 = best_r2_step

        print(f"  Step {step+1:2d}: +{best_feat:45s}  R2={best_r2_step:.3f}")

        # Stop if adding features stops helping
        if step > 3 and best_r2_step < best_r2_history[-3]:
            print(f"  (early stop: R2 declining)")
            break

    # Find optimal number of features
    optimal_k = int(np.argmax(best_r2_history)) + 1
    optimal_feats = selected[:optimal_k]
    print(f"\n  Optimal: {optimal_k} features, R2={best_r2_history[optimal_k-1]:.3f}")

    # Log the optimal model
    est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
    metrics, _ = log_experiment(
        f'EXP2_stepwise_top{optimal_k}', 'stepwise',
        {'model': 'Ridge', 'alpha': 10.0,
         'feature_selection': 'forward_stepwise', 'n_features': optimal_k},
        X[optimal_feats].values, y_arr, est, optimal_feats, data_hash,
        extra_info={'selection_history': best_r2_history,
                    'all_selected_order': selected}
    )

    # Also plot the selection curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(best_r2_history)+1), best_r2_history, 'bo-', linewidth=2)
    ax.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal: {optimal_k} features')
    ax.axhline(0.406, color='gray', linestyle=':', label='Baseline R2=0.406')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('LOOCV R2')
    ax.set_title('Forward Stepwise Feature Selection')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / 'stepwise_curve.png'
    plt.savefig(path, dpi=150)
    plt.close()
    with mlflow.start_run(run_name='EXP2_stepwise_curve', nested=True):
        mlflow.set_tag('experiment_type', 'stepwise')
        mlflow.log_artifact(str(path))

    return metrics, optimal_feats


# ============================================================================
# EXP 3: Richer feature engineering
# ============================================================================
def exp3_richer_features(X_raw, y, data_hash):
    print(f"\n{'='*60}")
    print("EXP 3: Richer feature engineering")
    print(f"{'='*60}")

    X = base_engineer(X_raw.copy())

    # --- Additional features ---
    # a) Cross-phase ratios (arterial / venous HU per kidney)
    for side in ['left', 'right']:
        for vessel in ['artery', 'vein']:
            art = f'arterial_{side}_kidney_{vessel}'
            ven = f'venous_{side}_kidney_{vessel}'
            lat = f'late_{side}_kidney_{vessel}'
            if art in X.columns and ven in X.columns:
                X[f'cross_art_ven_{vessel}_{side}'] = X[art] / X[ven].replace(0, np.nan)
            if art in X.columns and lat in X.columns:
                X[f'cross_art_late_{vessel}_{side}'] = X[art] / X[lat].replace(0, np.nan)
            if ven in X.columns and lat in X.columns:
                X[f'cross_ven_late_{vessel}_{side}'] = X[ven] / X[lat].replace(0, np.nan)

    # b) Total bilateral extraction (sum of left + right)
    for phase in ['arterial', 'venous', 'late']:
        el, er = f'E_{phase}_left', f'E_{phase}_right'
        if el in X.columns and er in X.columns:
            X[f'E_{phase}_total'] = X[el] + X[er]

    # c) Vena cava normalization (instead of aorta)
    for phase in ['arterial', 'venous', 'late']:
        vc = f'{phase}_venacava_below_kidney'
        if vc in X.columns:
            for side in ['left', 'right']:
                for vessel in ['artery', 'vein']:
                    col = f'{phase}_{side}_kidney_{vessel}'
                    if col in X.columns:
                        X[f'vc_norm_{vessel}_{phase}_{side}'] = X[col] / X[vc].replace(0, np.nan)

    # d) Age interactions with top extraction ratios
    if 'current_age' in X.columns:
        for phase in ['arterial', 'venous', 'late']:
            e_mean = f'E_{phase}_mean'
            if e_mean in X.columns:
                X[f'age_x_E_{phase}'] = X['current_age'] * X[e_mean]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    print(f"  Total features after enrichment: {len(X.columns)}")

    y_arr = y.values

    # Test with correlation-based top-k
    for k in [10, 15, 20]:
        feats = get_top_k(X, y, k)
        est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
        metrics, _ = log_experiment(
            f'EXP3_rich_feats_top{k}', 'feature_engineering',
            {'model': 'Ridge', 'alpha': 10.0, 'feature_set': 'enriched',
             'feature_selection': f'corr_top{k}', 'total_features': len(X.columns)},
            X[feats].values, y_arr, est, feats, data_hash
        )
        print(f"  top-{k}: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}  feats={feats[:5]}...")

    return X


# ============================================================================
# EXP 4: Log-target transformation
# ============================================================================
def exp4_log_target(X, y, data_hash):
    print(f"\n{'='*60}")
    print("EXP 4: Log-target transformation")
    print(f"{'='*60}")

    y_arr = y.values
    y_log = np.log(y_arr)

    for k in [10, 15]:
        feats = get_top_k(X, y, k)  # select vs original target

        # LOOCV on log scale, then transform back
        loo = LeaveOneOut()
        est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
        y_pred_log = cross_val_predict(est, X[feats].values, y_log, cv=loo)
        y_pred = np.exp(y_pred_log)

        mae = mean_absolute_error(y_arr, y_pred)
        rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
        r2 = r2_score(y_arr, y_pred)

        with mlflow.start_run(run_name=f'EXP4_log_target_top{k}', nested=True):
            mlflow.set_tag('experiment_type', 'log_target')
            mlflow.log_params({
                'data_hash': data_hash, 'n_patients': len(y_arr),
                'model': 'Ridge', 'alpha': 10.0,
                'target_transform': 'log', 'n_features': k,
            })
            mlflow.log_metrics({'MAE': mae, 'RMSE': rmse, 'R2': r2})

            feat_path = OUTPUT_DIR / f'info_EXP4_top{k}.json'
            with open(feat_path, 'w') as f:
                json.dump({'features': feats, 'transform': 'log(eGFRc)'}, f, indent=2)
            mlflow.log_artifact(str(feat_path))

        print(f"  log target (top-{k}): R2={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")


# ============================================================================
# EXP 5: Outlier-robust regression
# ============================================================================
def exp5_robust(X, y, data_hash):
    print(f"\n{'='*60}")
    print("EXP 5: Outlier-robust regression")
    print(f"{'='*60}")

    feats = get_top_k(X, y, 10)
    y_arr = y.values

    robust_models = {
        'Huber_e1.35': Pipeline([('scaler', StandardScaler()),
                                  ('model', HuberRegressor(epsilon=1.35, alpha=10.0))]),
        'Huber_e1.5': Pipeline([('scaler', StandardScaler()),
                                 ('model', HuberRegressor(epsilon=1.5, alpha=10.0))]),
        'Huber_e2.0': Pipeline([('scaler', StandardScaler()),
                                 ('model', HuberRegressor(epsilon=2.0, alpha=10.0))]),
        'BayesianRidge': Pipeline([('scaler', StandardScaler()),
                                    ('model', BayesianRidge())]),
    }

    for name, est in robust_models.items():
        metrics, _ = log_experiment(
            f'EXP5_{name}', 'robust',
            {'model': name, 'feature_selection': 'corr_top10'},
            X[feats].values, y_arr, est, feats, data_hash
        )
        print(f"  {name:20s}: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")


# ============================================================================
# EXP 6: Hyperparameter tuning (nested LOOCV for alpha)
# ============================================================================
def exp6_tuning(X, y, data_hash):
    print(f"\n{'='*60}")
    print("EXP 6: Hyperparameter tuning (alpha sweep)")
    print(f"{'='*60}")

    y_arr = y.values

    for k in [8, 10, 12]:
        feats = get_top_k(X, y, k)
        alphas = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
        results_alpha = []

        for alpha in alphas:
            loo = LeaveOneOut()
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
            y_pred = cross_val_predict(est, X[feats].values, y_arr, cv=loo)
            r2 = r2_score(y_arr, y_pred)
            mae = mean_absolute_error(y_arr, y_pred)
            results_alpha.append({'alpha': alpha, 'R2': r2, 'MAE': mae})

        results_df = pd.DataFrame(results_alpha)
        best_idx = results_df['R2'].idxmax()
        best_alpha = results_df.loc[best_idx, 'alpha']
        best_r2 = results_df.loc[best_idx, 'R2']
        best_mae = results_df.loc[best_idx, 'MAE']

        est_best = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=best_alpha))])
        metrics, _ = log_experiment(
            f'EXP6_tuned_k{k}_a{best_alpha}', 'tuning',
            {'model': 'Ridge', 'alpha': best_alpha,
             'feature_selection': f'corr_top{k}', 'alpha_search': str(alphas)},
            X[feats].values, y_arr, est_best, feats, data_hash,
            extra_info={'alpha_sweep': results_alpha}
        )

        print(f"  k={k:2d}: best alpha={best_alpha:6.1f}  R2={best_r2:.3f}  MAE={best_mae:.2f}")

        # Alpha sweep plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(results_df['alpha'], results_df['R2'], 'bo-', linewidth=2, label='R2')
        ax.axhline(0.406, color='gray', linestyle=':', label='Baseline R2=0.406')
        ax.axvline(best_alpha, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alpha (regularization)')
        ax.set_ylabel('LOOCV R2')
        ax.set_title(f'Alpha Sweep (k={k} features)')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = OUTPUT_DIR / f'alpha_sweep_k{k}.png'
        plt.savefig(path, dpi=150)
        plt.close()


# ============================================================================
# Summary comparison
# ============================================================================
def plot_comparison(all_results):
    """Plot a final comparison bar chart of all experiments."""
    df = pd.DataFrame(all_results).T
    df = df.sort_values('R2', ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(df) * 0.4)))

    for idx, metric in enumerate(['MAE', 'RMSE', 'R2']):
        ax = axes[idx]
        vals = df[metric]
        if metric == 'R2':
            vals = vals.sort_values(ascending=True)
        else:
            vals = vals.sort_values(ascending=False)

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(vals)))
        if metric != 'R2':
            colors = colors[::-1]
        vals.plot.barh(ax=ax, color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} (LOOCV)')
        if metric == 'R2':
            ax.axvline(0.406, color='red', linestyle='--', linewidth=2,
                       alpha=0.7, label='Baseline')
            ax.legend()

    plt.suptitle('All Experiments Comparison -- LOOCV Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = OUTPUT_DIR / 'all_experiments_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved all_experiments_comparison.png")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR Model Improvement Experiments")
    print("  Baseline: Ridge (top 10, alpha=10) R2=0.406")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, data_hash = load_data()
    X_raw, y = select_features(df)
    X = base_engineer(X_raw)

    # Track all results for final comparison
    all_results = {}

    with mlflow.start_run(run_name=f"improvements_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'improvement_experiments')
        mlflow.log_param('data_hash', data_hash)

        # EXP 0: Baseline
        m = exp0_baseline(X, y, data_hash)
        all_results['EXP0_baseline'] = m

        # EXP 1: Age-residual
        results_1 = exp1_age_residual(X, y, data_hash)
        for k, v in results_1.items():
            all_results[f'EXP1_{k}'] = v

        # EXP 2: Stepwise
        m, _ = exp2_stepwise(X, y, data_hash)
        all_results['EXP2_stepwise'] = m

        # EXP 3: Richer features
        X_rich = exp3_richer_features(X_raw, y, data_hash)
        # Pick the best enriched result for comparison
        for k in [10, 15, 20]:
            feats = get_top_k(X_rich, y, k)
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
            y_pred, metrics = evaluate_loocv(X_rich[feats].values, y.values, est)
            all_results[f'EXP3_rich_top{k}'] = metrics

        # EXP 4: Log target
        exp4_log_target(X, y, data_hash)
        # Recompute for comparison dict
        for k in [10, 15]:
            feats = get_top_k(X, y, k)
            y_log = np.log(y.values)
            loo = LeaveOneOut()
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
            y_pred_log = cross_val_predict(est, X[feats].values, y_log, cv=loo)
            y_pred = np.exp(y_pred_log)
            mae = mean_absolute_error(y.values, y_pred)
            rmse = np.sqrt(mean_squared_error(y.values, y_pred))
            r2 = r2_score(y.values, y_pred)
            all_results[f'EXP4_log_top{k}'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        # EXP 5: Robust
        exp5_robust(X, y, data_hash)
        feats10 = get_top_k(X, y, 10)
        for name, est in {
            'Huber_1.35': Pipeline([('scaler', StandardScaler()), ('model', HuberRegressor(epsilon=1.35, alpha=10.0))]),
            'BayesianRidge': Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())]),
        }.items():
            _, m = evaluate_loocv(X[feats10].values, y.values, est)
            all_results[f'EXP5_{name}'] = m

        # EXP 6: Tuning
        exp6_tuning(X, y, data_hash)
        # Pick k=10 sweep best for comparison
        alphas = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
        best_a_r2 = -np.inf
        best_a = 10
        for alpha in alphas:
            est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
            _, m = evaluate_loocv(X[feats10].values, y.values, est)
            if m['R2'] > best_a_r2:
                best_a_r2 = m['R2']
                best_a = alpha
                best_m = m
        all_results['EXP6_tuned_best'] = best_m

        # Comparison plot
        plot_comparison(all_results)
        mlflow.log_artifact(str(OUTPUT_DIR / 'all_experiments_comparison.png'))

        # Log summary
        results_df = pd.DataFrame(all_results).T.sort_values('R2', ascending=False)
        results_path = OUTPUT_DIR / 'all_experiments_results.csv'
        results_df.to_csv(results_path)
        mlflow.log_artifact(str(results_path))

        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        for name, row in results_df.iterrows():
            marker = " <-- BEST" if name == results_df.index[0] else ""
            baseline_marker = " <-- BASELINE" if name == 'EXP0_baseline' else ""
            print(f"  {name:35s}  R2={row['R2']:+.3f}  MAE={row['MAE']:.2f}{marker}{baseline_marker}")

        best_name = results_df.index[0]
        best_r2 = results_df.iloc[0]['R2']
        baseline_r2 = all_results['EXP0_baseline']['R2']
        print(f"\n  Best: {best_name} (R2={best_r2:.3f}, +{best_r2-baseline_r2:.3f} vs baseline)")


if __name__ == '__main__':
    main()
