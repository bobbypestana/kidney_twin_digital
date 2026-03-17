"""
vGFR Model Improvement Experiments -- Round 2
Baseline: Ridge stepwise (11 features), R2=0.533, MAE=7.33

Experiments:
  A: Combine winning strategies (stepwise + log-target, stepwise + alpha-tuning, stepwise + BayesianRidge)
  B: New model families (Gaussian Process, SVR, Stacking ensemble)
  C: Bronze table features (volumes, HU distributions)
  D: Per-kidney sum-constrained model
  E: Uncertainty quantification (GP prediction intervals)

Usage:
    python vgfr_improvements_r2.py
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
from plot_egfrc_vs_vgfr import plot_egfrc_vs_vgfr
from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
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

STEPWISE_FEATURES = [
    'current_age', 'arterial_venecava_between_kidney_hepatic',
    'E_late_right', 'norm_vein_arterial_right', 'arterial_portal_vein',
    'LR_ratio_vein_venous', 'mean_artery_venous',
    'late_venacava_above_hepatic', 'late_left_hepatic_vein',
    'E_arterial_right', 'norm_artery_venous_right',
]


# ============================================================================
# Data Loading
# ============================================================================
def load_gold():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').fetchdf()
    conn.close()
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]
    return df, data_hash


def load_bronze_features():
    """Load volume & HU distribution features from bronze segmentation tables."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    bronze_features = {}

    for phase in ['arterial', 'venous', 'late']:
        table = f'bronze.{phase}_segmentation'
        df = conn.execute(f'SELECT * FROM {table}').fetchdf()

        # Pivot: one row per case, columns = segment metrics
        for _, row in df.iterrows():
            case = row['case_number']
            seg = row['Segment']

            # Volume features
            case_str = str(case)
            if case_str not in bronze_features:
                bronze_features[case_str] = {}
            bronze_features[case_str][f'{seg}_vol_cm3'] = row['Volume cm3 (LM)']
            bronze_features[case_str][f'{seg}_voxels'] = row['Voxel count (LM)']
            # HU distribution features
            bronze_features[case_str][f'{seg}_hu_mean'] = row['Mean']
            bronze_features[case_str][f'{seg}_hu_std'] = row['Standard deviation']
            bronze_features[case_str][f'{seg}_hu_median'] = row['Median']
            bronze_features[case_str][f'{seg}_hu_p5'] = row['Percentile 5']
            bronze_features[case_str][f'{seg}_hu_p95'] = row['Percentile 95']

    conn.close()

    bronze_df = pd.DataFrame.from_dict(bronze_features, orient='index')
    bronze_df.index.name = 'case_number'
    bronze_df = bronze_df.apply(pd.to_numeric, errors='coerce')
    bronze_df = bronze_df.replace([np.inf, -np.inf], np.nan).fillna(bronze_df.median())
    return bronze_df


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
    X = X.copy()
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
    return X


# ============================================================================
# Helpers
# ============================================================================
def evaluate_loocv(X_arr, y_arr, estimator):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }


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


def get_top_k(X, y, k):
    return X.corrwith(y).abs().sort_values(ascending=False).index[:k].tolist()




# ============================================================================
# A: Combine winning strategies
# ============================================================================
def exp_A(X, y, data_hash):
    print(f"\n{'='*60}\nEXP A: Combine winning strategies\n{'='*60}")
    feats = [f for f in STEPWISE_FEATURES if f in X.columns]
    y_arr = y.values
    results = {}

    # A1: Stepwise + alpha tuning
    best_r2, best_a = -np.inf, 10
    for alpha in [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 100]:
        est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
        _, m = evaluate_loocv(X[feats].values, y_arr, est)
        if m['R2'] > best_r2:
            best_r2, best_a, best_m = m['R2'], alpha, m
    est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=best_a))])
    y_pred, metrics = evaluate_loocv(X[feats].values, y_arr, est)
    log_run('A1_stepwise_tuned', 'combine', {'model': 'Ridge', 'alpha': best_a,
            'features': 'stepwise', 'n_features': len(feats)}, metrics, feats, data_hash)
    results['A1_stepwise_tuned'] = {
        'metrics': metrics,
        'y_pred': y_pred,
        'features': feats,
        'model_name': 'Ridge_StepwiseTuned'
    }
    print(f"  A1) Stepwise + alpha={best_a}: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    # A2: Stepwise + log-target
    y_log = np.log(y_arr)
    loo = LeaveOneOut()
    est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=best_a))])
    y_pred_log = cross_val_predict(est, X[feats].values, y_log, cv=loo)
    y_pred = np.exp(y_pred_log)
    metrics = {'MAE': mean_absolute_error(y_arr, y_pred),
               'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
               'R2': r2_score(y_arr, y_pred)}
    log_run('A2_stepwise_log', 'combine', {'model': 'Ridge', 'alpha': best_a,
            'features': 'stepwise', 'target_transform': 'log'}, metrics, feats, data_hash)
    results['A2_stepwise_log'] = {
        'metrics': metrics,
        'y_pred': y_pred,
        'features': feats,
        'model_name': 'Ridge_StepwiseLog'
    }
    print(f"  A2) Stepwise + log-target: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    # A3: Stepwise + BayesianRidge
    est = Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())])
    y_pred, metrics = evaluate_loocv(X[feats].values, y_arr, est)
    log_run('A3_stepwise_bayesian', 'combine', {'model': 'BayesianRidge',
            'features': 'stepwise'}, metrics, feats, data_hash)
    results['A3_stepwise_bayesian'] = {
        'metrics': metrics,
        'y_pred': y_pred,
        'features': feats,
        'model_name': 'BayesianRidge_Stepwise'
    }
    print(f"  A3) Stepwise + BayesianRidge: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    return results


# ============================================================================
# B: New model families
# ============================================================================
def exp_B(X, y, data_hash):
    print(f"\n{'='*60}\nEXP B: New model families\n{'='*60}")
    feats = [f for f in STEPWISE_FEATURES if f in X.columns]
    y_arr = y.values
    results = {}

    # B1: Gaussian Process Regression
    for kernel_name, kernel in [
        ('RBF', ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)),
        ('Matern', ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)),
    ]:
        est = Pipeline([('scaler', StandardScaler()),
                        ('model', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42))])
        y_pred, metrics = evaluate_loocv(X[feats].values, y_arr, est)
        name = f'B1_GP_{kernel_name}'
        log_run(name, 'new_model', {'model': f'GP_{kernel_name}', 'features': 'stepwise'},
                metrics, feats, data_hash)
        results[name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'features': feats,
            'model_name': f'GP_{kernel_name}'
        }
        print(f"  {name}: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    # B2: SVR
    for C_val in [1, 10, 100]:
        est = Pipeline([('scaler', StandardScaler()),
                        ('model', SVR(kernel='rbf', C=C_val, epsilon=0.1))])
        y_pred, metrics = evaluate_loocv(X[feats].values, y_arr, est)
        name = f'B2_SVR_C{C_val}'
        log_run(name, 'new_model', {'model': 'SVR', 'C': C_val, 'features': 'stepwise'},
                metrics, feats, data_hash)
        results[name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'features': feats,
            'model_name': f'SVR_C{C_val}'
        }
        print(f"  {name}: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    # B3: Stacking ensemble
    estimators = [
        ('ridge', Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])),
        ('bayesian', Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())])),
        ('svr', Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=10))])),
    ]
    stack = StackingRegressor(estimators=estimators,
                              final_estimator=Ridge(alpha=1.0), cv=5)
    est = Pipeline([('scaler', StandardScaler()), ('model', stack)])
    y_pred, metrics = evaluate_loocv(X[feats].values, y_arr, est)
    name = 'B3_Stacking'
    log_run(name, 'new_model', {'model': 'Stacking_Ridge_Bayes_SVR', 'features': 'stepwise'},
            metrics, feats, data_hash)
    results[name] = {
        'metrics': metrics,
        'y_pred': y_pred,
        'features': feats,
        'model_name': 'Stacking_Ridge_Bayes_SVR'
    }
    print(f"  {name}: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    return results


# ============================================================================
# C: Bronze table features
# ============================================================================
def exp_C(X_gold, y, data_hash):
    print(f"\n{'='*60}\nEXP C: Bronze table features (volumes + HU distributions)\n{'='*60}")
    y_arr = y.values
    results = {}

    bronze_df = load_bronze_features()
    # Match gold record_id to bronze case_number
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    gold_df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').fetchdf()
    conn.close()

    record_ids = gold_df['record_id'].values
    # Bronze uses case_number (str), gold uses record_id (int)
    available_cases = set(bronze_df.index.tolist())

    # Build combined feature matrix
    gold_feats = [f for f in STEPWISE_FEATURES if f in X_gold.columns]
    X_combined_list = []
    valid_idx = []

    for i, rid in enumerate(record_ids):
        rid_str = str(int(rid))
        if rid_str in available_cases:
            gold_row = X_gold.iloc[i][gold_feats].values
            bronze_row = bronze_df.loc[rid_str].values
            combined = np.concatenate([gold_row, bronze_row])
            X_combined_list.append(combined)
            valid_idx.append(i)

    if len(X_combined_list) == 0:
        print("  WARNING: No matching cases between gold and bronze tables")
        return results

    X_combined = np.array(X_combined_list)
    y_matched = y_arr[valid_idx]
    bronze_feat_names = gold_feats + list(bronze_df.columns)

    # Handle NaN/Inf
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Matched {len(valid_idx)}/{len(record_ids)} cases, {X_combined.shape[1]} features")

    # C1: Stepwise gold + all bronze
    est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=20.0))])
    y_pred, metrics = evaluate_loocv(X_combined, y_matched, est)
    name = 'C1_gold_plus_bronze'
    log_run(name, 'bronze', {'model': 'Ridge', 'alpha': 20.0,
            'n_gold_features': len(gold_feats), 'n_bronze_features': len(bronze_df.columns),
            'n_patients_matched': len(valid_idx)}, metrics, bronze_feat_names[:20], data_hash)
    results[name] = {
        'metrics': metrics,
        'y_pred': y_pred,
        'features': bronze_feat_names[:20], # Truncated for display
        'model_name': 'Ridge_GoldAndBronze'
    }
    print(f"  C1) Gold + all bronze: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    # C2: Volume-only features from bronze
    vol_cols = [c for c in bronze_df.columns if 'vol_cm3' in c]
    if vol_cols:
        X_vol = np.array([bronze_df.loc[str(int(record_ids[i]))][vol_cols].values for i in valid_idx])
        X_vol = np.nan_to_num(X_vol, nan=0.0, posinf=0.0, neginf=0.0)
        X_gold_vol = np.hstack([X_gold.iloc[valid_idx][gold_feats].values, X_vol])

        est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=20.0))])
        y_pred, metrics = evaluate_loocv(X_gold_vol, y_matched, est)
        name = 'C2_gold_plus_volumes'
        log_run(name, 'bronze', {'model': 'Ridge', 'alpha': 20.0,
                'features': 'stepwise + volumes'}, metrics, gold_feats + vol_cols[:10], data_hash)
        results[name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'features': gold_feats + vol_cols[:10],
            'model_name': 'Ridge_GoldAndVolumes'
        }
        print(f"  C2) Gold + volumes only: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}  ({len(vol_cols)} vol features)")

    # C3: Stepwise on bronze features only (feature selection)
    X_bronze_only = np.array([bronze_df.loc[str(int(record_ids[i]))].values for i in valid_idx])
    X_bronze_only = np.nan_to_num(X_bronze_only, nan=0.0, posinf=0.0, neginf=0.0)
    bronze_series = {c: X_bronze_only[:, j] for j, c in enumerate(bronze_df.columns)}
    bronze_corrs = {c: abs(np.corrcoef(v, y_matched)[0,1]) for c, v in bronze_series.items()
                    if not np.isnan(np.corrcoef(v, y_matched)[0,1])}
    top_bronze = sorted(bronze_corrs, key=bronze_corrs.get, reverse=True)[:15]

    top_bronze_idx = [list(bronze_df.columns).index(c) for c in top_bronze]
    X_top_bronze = X_bronze_only[:, top_bronze_idx]
    X_gold_top_bronze = np.hstack([X_gold.iloc[valid_idx][gold_feats].values, X_top_bronze])

    est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=20.0))])
    y_pred, metrics = evaluate_loocv(X_gold_top_bronze, y_matched, est)
    name = 'C3_gold_plus_top_bronze'
    log_run(name, 'bronze', {'model': 'Ridge', 'alpha': 20.0,
            'features': 'stepwise + top15 bronze'}, metrics,
            gold_feats + top_bronze[:10], data_hash)
    results[name] = {
        'metrics': metrics,
        'y_pred': y_pred,
        'features': gold_feats + top_bronze[:10],
        'model_name': 'Ridge_GoldAndTopBronze'
    }
    print(f"  C3) Gold + top-15 bronze: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")
    print(f"      Top bronze features: {top_bronze[:5]}")

    return results


# ============================================================================
# D: Per-kidney sum-constrained model
# ============================================================================
def exp_D(X, y, df, data_hash):
    print(f"\n{'='*60}\nEXP D: Per-kidney sum-constrained model\n{'='*60}")
    y_arr = y.values
    results = {}

    # Left kidney features
    left_feats = [c for c in X.columns if 'left' in c.lower() and 'ratio' not in c.lower()]
    right_feats = [c for c in X.columns if 'right' in c.lower() and 'ratio' not in c.lower()]

    # Add age to both
    shared = ['current_age']
    left_feats = shared + [f for f in left_feats if f not in shared]
    right_feats = shared + [f for f in right_feats if f not in shared]

    print(f"  Left features:  {len(left_feats)}")
    print(f"  Right features: {len(right_feats)}")

    # Model: predict eGFRc/2 for each kidney, then sum
    # Assumption: each kidney contributes ~half of total eGFRc
    y_half = y_arr / 2.0

    loo = LeaveOneOut()
    y_combined = np.zeros(len(y_arr))

    for train_idx, test_idx in loo.split(X.values):
        # Left kidney model
        left_model = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=20.0))])
        left_model.fit(X[left_feats].values[train_idx], y_half[train_idx])
        left_pred = left_model.predict(X[left_feats].values[test_idx])

        # Right kidney model
        right_model = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=20.0))])
        right_model.fit(X[right_feats].values[train_idx], y_half[train_idx])
        right_pred = right_model.predict(X[right_feats].values[test_idx])

        y_combined[test_idx] = left_pred + right_pred

    metrics = {
        'MAE': mean_absolute_error(y_arr, y_combined),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_combined)),
        'R2': r2_score(y_arr, y_combined),
    }
    name = 'D1_per_kidney_sum'
    log_run(name, 'per_kidney', {'model': 'Ridge_per_kidney', 'alpha': 20.0,
            'n_left_feats': len(left_feats), 'n_right_feats': len(right_feats)},
            metrics, left_feats[:5] + right_feats[:5], data_hash)
    results[name] = {
        'metrics': metrics,
        'y_pred': y_combined,
        'features': left_feats[:5] + right_feats[:5],
        'model_name': 'Ridge_PerKidneySum'
    }
    print(f"  D1) Per-kidney sum: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")

    # D2: Weighted per-kidney (learn the split ratio too)
    # Use left features + right features together but structured
    combined_feats = list(set(left_feats + right_feats))
    est = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=20.0))])
    y_pred, metrics2 = evaluate_loocv(X[combined_feats].values, y_arr, est)
    name2 = 'D2_bilateral_combined'
    log_run(name2, 'per_kidney', {'model': 'Ridge_bilateral', 'alpha': 20.0,
            'n_features': len(combined_feats)}, metrics2, combined_feats[:10], data_hash)
    results[name2] = {
        'metrics': metrics2,
        'y_pred': y_pred,
        'features': combined_feats[:10],
        'model_name': 'Ridge_BilateralCombined'
    }
    print(f"  D2) Bilateral combined: R2={metrics2['R2']:.3f}  MAE={metrics2['MAE']:.2f}")

    return results


# ============================================================================
# E: Uncertainty quantification
# ============================================================================
def exp_E(X, y, data_hash):
    print(f"\n{'='*60}\nEXP E: Uncertainty quantification\n{'='*60}")
    feats = [f for f in STEPWISE_FEATURES if f in X.columns]
    y_arr = y.values
    results = {}

    # E1: GP with prediction intervals
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    loo = LeaveOneOut()
    y_pred_gp = np.zeros(len(y_arr))
    y_std_gp = np.zeros(len(y_arr))

    scaler = StandardScaler()
    for train_idx, test_idx in loo.split(X[feats].values):
        X_train = scaler.fit_transform(X[feats].values[train_idx])
        X_test = scaler.transform(X[feats].values[test_idx])
        gp = GaussianProcessRegressor(kernel=clone(kernel), n_restarts_optimizer=10, random_state=42)
        gp.fit(X_train, y_arr[train_idx])
        pred, std = gp.predict(X_test, return_std=True)
        y_pred_gp[test_idx] = pred
        y_std_gp[test_idx] = std

    metrics = {
        'MAE': mean_absolute_error(y_arr, y_pred_gp),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred_gp)),
        'R2': r2_score(y_arr, y_pred_gp),
    }
    results['E1_GP_uncertainty'] = {
        'metrics': metrics,
        'y_pred': y_pred_gp,
        'features': feats,
        'model_name': 'GP_Uncertainty'
    }

    # Coverage: what fraction of true values fall within 95% CI?
    ci_95_lower = y_pred_gp - 1.96 * y_std_gp
    ci_95_upper = y_pred_gp + 1.96 * y_std_gp
    coverage = np.mean((y_arr >= ci_95_lower) & (y_arr <= ci_95_upper))
    mean_ci_width = np.mean(2 * 1.96 * y_std_gp)

    print(f"  E1) GP uncertainty: R2={metrics['R2']:.3f}  MAE={metrics['MAE']:.2f}")
    print(f"      95% CI coverage: {coverage:.1%}")
    print(f"      Mean CI width:   {mean_ci_width:.1f} mL/min/1.73m2")

    with mlflow.start_run(run_name='E1_GP_uncertainty', nested=True):
        mlflow.set_tag('experiment_type', 'uncertainty')
        mlflow.log_params({'model': 'GP_Matern', 'features': 'stepwise', 'data_hash': data_hash})
        mlflow.log_metrics({**metrics, 'CI_95_coverage': coverage, 'CI_mean_width': mean_ci_width})

    # Plot: predictions with error bars
    fig, ax = plt.subplots(figsize=(9, 9))
    sort_idx = np.argsort(y_arr)

    ax.errorbar(y_arr[sort_idx], y_pred_gp[sort_idx], yerr=1.96*y_std_gp[sort_idx],
                fmt='o', markersize=8, capsize=4, color='steelblue', ecolor='lightcoral',
                alpha=0.8, label='Prediction +/- 95% CI')
    lims = [min(y_arr.min(), y_pred_gp.min()) - 10, max(y_arr.max(), y_pred_gp.max()) + 10]
    ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7, label='Identity')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('eGFRc (mL/min/1.73m$^2$) -- Creatinine-based', fontsize=12)
    ax.set_ylabel('vGFR (mL/min/1.73m$^2$) -- CT Predicted', fontsize=12)
    ax.set_title(f'GP Regression with 95% Prediction Intervals\n'
                 f'R2={metrics["R2"]:.3f}, Coverage={coverage:.0%}, CI width={mean_ci_width:.1f}',
                 fontsize=13)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
    plt.tight_layout()
    # p = OUTPUT_DIR / 'gp_uncertainty.png'
    # plt.savefig(p, dpi=200); plt.close()
    # with mlflow.start_run(run_name='E1_GP_plot', nested=True):
    #     mlflow.set_tag('experiment_type', 'uncertainty')
    #     mlflow.log_artifact(str(p))

    # E2: BayesianRidge uncertainty
    loo = LeaveOneOut()
    y_pred_br = np.zeros(len(y_arr))
    y_std_br = np.zeros(len(y_arr))

    for train_idx, test_idx in loo.split(X[feats].values):
        X_train = scaler.fit_transform(X[feats].values[train_idx])
        X_test = scaler.transform(X[feats].values[test_idx])
        br = BayesianRidge()
        br.fit(X_train, y_arr[train_idx])
        pred, std = br.predict(X_test, return_std=True)
        y_pred_br[test_idx] = pred
        y_std_br[test_idx] = std

    metrics_br = {
        'MAE': mean_absolute_error(y_arr, y_pred_br),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred_br)),
        'R2': r2_score(y_arr, y_pred_br),
    }
    ci_lower_br = y_pred_br - 1.96 * y_std_br
    ci_upper_br = y_pred_br + 1.96 * y_std_br
    coverage_br = np.mean((y_arr >= ci_lower_br) & (y_arr <= ci_upper_br))
    width_br = np.mean(2 * 1.96 * y_std_br)

    results['E2_BayesianRidge_uncertainty'] = {
        'metrics': metrics_br,
        'y_pred': y_pred_br,
        'features': feats,
        'model_name': 'BayesianRidge_Uncertainty'
    }
    print(f"  E2) BayesianRidge: R2={metrics_br['R2']:.3f}  MAE={metrics_br['MAE']:.2f}")
    print(f"      95% CI coverage: {coverage_br:.1%}, width: {width_br:.1f}")

    with mlflow.start_run(run_name='E2_BayesianRidge_uncertainty', nested=True):
        mlflow.set_tag('experiment_type', 'uncertainty')
        mlflow.log_params({'model': 'BayesianRidge', 'features': 'stepwise', 'data_hash': data_hash})
        mlflow.log_metrics({**metrics_br, 'CI_95_coverage': coverage_br, 'CI_mean_width': width_br})

    return results


# ============================================================================
# Summary
# ============================================================================
def plot_comparison(all_results):
    results_metrics = {k: v['metrics'] for k, v in all_results.items()}
    df = pd.DataFrame(results_metrics).T.sort_values('R2', ascending=True)
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
        ax.set_xlabel(metric); ax.set_title(f'{metric} (LOOCV)')
        if metric == 'R2':
            ax.axvline(0.533, color='red', linestyle='--', linewidth=2, alpha=0.7, label='R1 best=0.533')
            ax.axvline(0.406, color='gray', linestyle=':', alpha=0.5, label='Baseline=0.406')
            ax.legend()
    plt.suptitle('Round 2 Experiments -- LOOCV Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = OUTPUT_DIR / 'round2_comparison.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    return p


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR Round 2 Improvements")
    print("  R1 best: Stepwise Ridge R2=0.533, MAE=7.33")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, data_hash = load_gold()
    X_raw, y = select_features(df)
    X = base_engineer(X_raw)
    print(f"[OK] {len(X)} patients, {len(X.columns)} features")

    all_results = {}

    with mlflow.start_run(run_name=f"round2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag('run_type', 'round2_improvements')
        mlflow.log_param('data_hash', data_hash)

        # A: Combine strategies
        for k, v in exp_A(X, y, data_hash).items():
            all_results[k] = v

        # B: New models
        for k, v in exp_B(X, y, data_hash).items():
            all_results[k] = v

        # C: Bronze features
        for k, v in exp_C(X, y, data_hash).items():
            all_results[k] = v

        # D: Per-kidney
        for k, v in exp_D(X, y, df, data_hash).items():
            all_results[k] = v

        # E: Uncertainty
        for k, v in exp_E(X, y, data_hash).items():
            all_results[k] = v

        # Summary
        p = plot_comparison(all_results)
        mlflow.log_artifact(str(p))

        results_metrics = {k: v['metrics'] for k, v in all_results.items()}
        results_df = pd.DataFrame(results_metrics).T.sort_values('R2', ascending=False)
        results_df.to_csv(OUTPUT_DIR / 'round2_results.csv')
        mlflow.log_artifact(str(OUTPUT_DIR / 'round2_results.csv'))

        champion_name = results_df.index[0]
        champ_data = all_results[champion_name]
        champ_metrics = champ_data['metrics']
        champ_y_pred = champ_data['y_pred']
        champ_features = champ_data['features']
        model_label = f"R2 Champ - {champ_data['model_name']}"

        # 1. Main Scatter Plot
        plot_path = OUTPUT_DIR / f'round2_champion_{champ_data["model_name"]}.png'
        plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, champ_features, champ_metrics, plot_path)
        mlflow.log_artifact(str(plot_path))

        # 2. Residual Plot
        res_path = OUTPUT_DIR / f'round2_champion_{champ_data["model_name"]}_residuals.png'
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
        print("ROUND 2 RESULTS")
        print(f"{'='*60}")
        for name, row in results_df.iterrows():
            marker = " <-- BEST" if name == results_df.index[0] else ""
            print(f"  {name:35s}  R2={row['R2']:+.3f}  MAE={row['MAE']:.2f}{marker}")

        best_name = results_df.index[0]
        print(f"\n  Winner: {best_name} (R2={results_df.iloc[0]['R2']:.3f})")
        print(f"  vs R1 best (stepwise): {results_df.iloc[0]['R2'] - 0.533:+.3f}")


if __name__ == '__main__':
    main()
