"""
vGFR ML Model -- POC with MLflow Tracking
Predict eGFRc (kidney function) from CT scan segmentation features.
Goal: calibrate a kidney digital twin per patient using imaging data.

Tracks with MLflow:
  - Data version: hash, schema, row count, column list
  - Feature engineering details
  - Per-model: params, LOOCV metrics, predictions, artifacts
  - Feature correlations and importance rankings
  - Best/worst feature analysis

Usage:
    python vgfr_model.py                 # Run pipeline + log to MLflow
    mlflow ui --port 5000                # View results at localhost:5000
"""

import duckdb
import pandas as pd
import numpy as np
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from plot_egfrc_vs_vgfr import plot_egfrc_vs_vgfr
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data.duckdb'
OUTPUT_DIR = Path(__file__).parent / 'ml_results'
OUTPUT_DIR.mkdir(exist_ok=True)

MLFLOW_DIR = Path('C:/tmp/mlflow_vgfr')
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "vGFR_Kidney_Digital_Twin"

TARGET = 'egfrc'

# Columns to exclude
EXCLUDE_PREFIXES = ['vgfr_', 'conc_lit_', 'conc_late_', 'w_pv_', 'w_back_']
EXCLUDE_COLS = [
    'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
    'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days',
    TARGET,
]


# ============================================================================
# 1. Data Loading & Versioning
# ============================================================================
def load_data():
    """Load data from DuckDB and compute data fingerprint."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').fetchdf()
    conn.close()

    # Data fingerprint for versioning
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]

    data_info = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'data_hash': data_hash,
        'db_path': str(DB_PATH),
        'target_mean': float(df[TARGET].mean()),
        'target_std': float(df[TARGET].std()),
        'target_min': float(df[TARGET].min()),
        'target_max': float(df[TARGET].max()),
        'null_counts': df.isnull().sum().to_dict(),
    }

    print(f"[OK] Loaded {len(df)} patients, {len(df.columns)} columns")
    print(f"[OK] Data hash: {data_hash}")
    return df, data_info


# ============================================================================
# 2. Feature Selection & Engineering
# ============================================================================
def select_features(df):
    """Select features by excluding vgfr_*, conc_*, w_*, leakage cols."""
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

    print(f"[OK] Selected {len(X.columns)} raw features")
    return X, y


def engineer_features(X):
    """Create derived features from raw HU measurements."""
    X = X.copy()
    engineered_names = []

    # --- Extraction ratios: (artery - vein) / artery ---
    for phase, side in [('arterial','left'),('arterial','right'),
                        ('venous','left'),('venous','right'),
                        ('late','left'),('late','right')]:
        art = f'{phase}_{side}_kidney_artery'
        vein = f'{phase}_{side}_kidney_vein'
        if art in X.columns and vein in X.columns:
            name = f'E_{phase}_{side}'
            X[name] = (X[art] - X[vein]) / X[art].replace(0, np.nan)
            engineered_names.append(name)

    # --- Left-right symmetry ---
    for phase in ['arterial', 'venous', 'late']:
        la, ra = f'{phase}_left_kidney_artery', f'{phase}_right_kidney_artery'
        lv, rv = f'{phase}_left_kidney_vein', f'{phase}_right_kidney_vein'
        if la in X.columns and ra in X.columns:
            name = f'LR_ratio_artery_{phase}'
            X[name] = X[la] / X[ra].replace(0, np.nan)
            engineered_names.append(name)
        if lv in X.columns and rv in X.columns:
            name = f'LR_ratio_vein_{phase}'
            X[name] = X[lv] / X[rv].replace(0, np.nan)
            engineered_names.append(name)

    # --- Phase contrast (arterial vs late) ---
    for side in ['left', 'right']:
        for vessel in ['artery', 'vein']:
            art_col = f'arterial_{side}_kidney_{vessel}'
            late_col = f'late_{side}_kidney_{vessel}'
            if art_col in X.columns and late_col in X.columns:
                name = f'phase_contrast_{vessel}_{side}'
                X[name] = X[art_col] - X[late_col]
                engineered_names.append(name)

    # --- Kidney-to-aorta normalization ---
    for phase in ['arterial', 'venous', 'late']:
        aorta = f'{phase}_aorta'
        if aorta in X.columns:
            for side in ['left', 'right']:
                for vessel in ['artery', 'vein']:
                    col = f'{phase}_{side}_kidney_{vessel}'
                    if col in X.columns:
                        name = f'norm_{vessel}_{phase}_{side}'
                        X[name] = X[col] / X[aorta].replace(0, np.nan)
                        engineered_names.append(name)

    # --- Bilateral means ---
    for phase in ['arterial', 'venous', 'late']:
        la, ra = f'{phase}_left_kidney_artery', f'{phase}_right_kidney_artery'
        lv, rv = f'{phase}_left_kidney_vein', f'{phase}_right_kidney_vein'
        if la in X.columns and ra in X.columns:
            name = f'mean_artery_{phase}'
            X[name] = (X[la] + X[ra]) / 2
            engineered_names.append(name)
        if lv in X.columns and rv in X.columns:
            name = f'mean_vein_{phase}'
            X[name] = (X[lv] + X[rv]) / 2
            engineered_names.append(name)

        el, er = f'E_{phase}_left', f'E_{phase}_right'
        if el in X.columns and er in X.columns:
            name = f'E_{phase}_mean'
            X[name] = (X[el] + X[er]) / 2
            engineered_names.append(name)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    print(f"[OK] Engineered {len(engineered_names)} new features -> {len(X.columns)} total")
    return X, engineered_names


# ============================================================================
# 3. Feature Analysis
# ============================================================================
def analyze_features(X, y):
    """Compute correlations and rank features."""
    correlations = X.corrwith(y).sort_values(ascending=False)

    # Feature ranking table
    feat_analysis = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values,
        'abs_correlation': correlations.abs().values,
        'mean': X[correlations.index].mean().values,
        'std': X[correlations.index].std().values,
        'null_count': X[correlations.index].isnull().sum().values,
    }).sort_values('abs_correlation', ascending=False).reset_index(drop=True)

    feat_analysis['rank'] = range(1, len(feat_analysis) + 1)

    print(f"\n{'='*60}")
    print("TOP 10 FEATURES (by |correlation| with eGFRc)")
    print(f"{'='*60}")
    for _, row in feat_analysis.head(10).iterrows():
        print(f"  #{row['rank']:2d}  {row['feature']:45s}  r={row['correlation']:+.3f}")

    print(f"\nBOTTOM 5 FEATURES:")
    for _, row in feat_analysis.tail(5).iterrows():
        print(f"  #{row['rank']:2d}  {row['feature']:45s}  r={row['correlation']:+.3f}")

    return correlations, feat_analysis


def plot_eda(X, y, correlations, feat_analysis):
    """Generate and save EDA plots. Returns dict of plot paths."""
    plots = {}

    # --- Correlation bar chart ---
    fig, ax = plt.subplots(figsize=(10, 8))
    top_all = pd.concat([correlations.head(10), correlations.tail(10)])
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_all.values]
    top_all.plot.barh(ax=ax, color=colors)
    ax.set_xlabel('Pearson Correlation with eGFRc')
    ax.set_title('Top 20 Feature Correlations with eGFRc')
    ax.axvline(0, color='black', linewidth=0.5)
    # plt.tight_layout()
    # path = OUTPUT_DIR / 'correlation_top20.png'
    # plt.savefig(path, dpi=150)
    # plt.close()
    # plots['correlation_top20'] = str(path)

    # --- Target distribution ---
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.hist(y, bins=12, edgecolor='black', alpha=0.7, color='steelblue')
    # ax.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.1f}')
    # ax.set_xlabel('eGFRc (mL/min/1.73m2)')
    # ax.set_ylabel('Count')
    # ax.set_title('Target Distribution: eGFRc')
    # ax.legend()
    # plt.tight_layout()
    # path = OUTPUT_DIR / 'target_distribution.png'
    # plt.savefig(path, dpi=150)
    # plt.close()
    # plots['target_distribution'] = str(path)

    # --- Correlation heatmap ---
    # top_feats = feat_analysis.head(12)['feature'].tolist()
    # subset = X[top_feats].copy()
    # subset[TARGET] = y.values
    # corr_matrix = subset.corr()

    # fig, ax = plt.subplots(figsize=(14, 12))
    # sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
    #             square=True, ax=ax, vmin=-1, vmax=1)
    # ax.set_title('Correlation Heatmap: Top Features + eGFRc')
    # plt.tight_layout()
    # path = OUTPUT_DIR / 'correlation_heatmap.png'
    # plt.savefig(path, dpi=150)
    # plt.close()
    # plots['correlation_heatmap'] = str(path)

    return plots


# ============================================================================
# 4. Model Training with LOOCV + MLflow
# ============================================================================
def define_models():
    """Define all model configurations to test."""
    return {
        'Linear_top5': {
            'estimator': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
            'k': 5, 'family': 'linear', 'params': {'type': 'OLS', 'regularization': 'none'}
        },
        'Linear_top10': {
            'estimator': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
            'k': 10, 'family': 'linear', 'params': {'type': 'OLS', 'regularization': 'none'}
        },
        'Ridge_top10_a1': {
            'estimator': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
            'k': 10, 'family': 'linear', 'params': {'type': 'Ridge', 'alpha': 1.0}
        },
        'Ridge_top10_a10': {
            'estimator': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))]),
            'k': 10, 'family': 'linear', 'params': {'type': 'Ridge', 'alpha': 10.0}
        },
        'Ridge_top15_a10': {
            'estimator': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))]),
            'k': 15, 'family': 'linear', 'params': {'type': 'Ridge', 'alpha': 10.0}
        },
        'Lasso_top10_a1': {
            'estimator': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=1.0, max_iter=10000))]),
            'k': 10, 'family': 'linear', 'params': {'type': 'Lasso', 'alpha': 1.0}
        },
        'ElasticNet_top10': {
            'estimator': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000))]),
            'k': 10, 'family': 'linear', 'params': {'type': 'ElasticNet', 'alpha': 1.0, 'l1_ratio': 0.5}
        },
        'RF_top10_d3': {
            'estimator': Pipeline([('model', RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42))]),
            'k': 10, 'family': 'tree', 'params': {'type': 'RandomForest', 'n_estimators': 100, 'max_depth': 3}
        },
        'RF_top15_d4': {
            'estimator': Pipeline([('model', RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42))]),
            'k': 15, 'family': 'tree', 'params': {'type': 'RandomForest', 'n_estimators': 100, 'max_depth': 4}
        },
        'GBR_top10': {
            'estimator': Pipeline([('model', GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42))]),
            'k': 10, 'family': 'tree', 'params': {'type': 'GradientBoosting', 'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.1}
        },
    }


def train_and_evaluate(X, y, correlations, feat_analysis, data_info, eda_plots):
    """Train all models with LOOCV, log everything to MLflow."""
    loo = LeaveOneOut()
    sorted_feats = correlations.abs().sort_values(ascending=False).index.tolist()
    models = define_models()

    all_results = {}
    all_predictions = {}

    print(f"\n{'='*60}")
    print("MODEL EVALUATION (LOOCV) + MLflow Logging")
    print(f"{'='*60}")

    for model_name, config in models.items():
        k = config['k']
        top_k_feats = sorted_feats[:k]
        X_sel = X[top_k_feats].values
        y_arr = y.values

        # LOOCV predictions
        y_pred = cross_val_predict(config['estimator'], X_sel, y_arr, cv=loo)

        mae = mean_absolute_error(y_arr, y_pred)
        rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
        r2 = r2_score(y_arr, y_pred)
        max_error = float(np.max(np.abs(y_arr - y_pred)))
        median_ae = float(np.median(np.abs(y_arr - y_pred)))

        # Fit on full data for feature importance
        config['estimator'].fit(X_sel, y_arr)

        # --- MLflow child run for each model ---
        with mlflow.start_run(run_name=model_name, nested=True):
            # Data tracking
            mlflow.log_params({
                'data_hash': data_info['data_hash'],
                'n_patients': data_info['n_rows'],
                'n_raw_columns': data_info['n_columns'],
                'n_features_total': len(X.columns),
                'n_features_selected': k,
                'cv_method': 'LOOCV',
            })

            # Model params
            for pk, pv in config['params'].items():
                mlflow.log_param(f'model_{pk}', pv)
            mlflow.log_param('model_family', config['family'])

            # Features used (as a JSON artifact)
            feat_info = {
                'features_used': top_k_feats,
                'feature_correlations': {f: float(correlations[f]) for f in top_k_feats},
                'feature_ranks': {f: int(feat_analysis[feat_analysis['feature']==f]['rank'].values[0])
                                  for f in top_k_feats if f in feat_analysis['feature'].values},
            }

            # Feature importance from fitted model
            if hasattr(config['estimator'][-1], 'feature_importances_'):
                imp = config['estimator'][-1].feature_importances_
                feat_info['rf_importance'] = {f: float(imp[i]) for i, f in enumerate(top_k_feats)}
            elif hasattr(config['estimator'][-1], 'coef_'):
                coefs = config['estimator'][-1].coef_
                feat_info['model_coefficients'] = {f: float(coefs[i]) for i, f in enumerate(top_k_feats)}

            # Metrics
            mlflow.log_metrics({
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'max_error': max_error,
                'median_absolute_error': median_ae,
            })

            # Per-feature correlations are tracked in the features JSON artifact above

            # Log the sklearn model (optional, removed for lean run)
            # mlflow.sklearn.log_model(config['estimator'], name=f"model_{model_name}")

        all_results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2,
                                   'max_error': max_error, 'median_ae': median_ae, 'k': k}
        all_predictions[model_name] = y_pred

        print(f"  {model_name:30s}  MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:+6.3f}")

    return all_results, all_predictions


# ============================================================================
# 5. Summary Plots & Best Model Analysis
# ============================================================================
def plot_summary(results, predictions, y, X, correlations):
    """Generate comparison plots and best-model analysis."""
    results_df = pd.DataFrame(results).T
    plots = {}

    # --- Model comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for idx, metric in enumerate(['MAE', 'RMSE', 'R2']):
        ax = axes[idx]
        vals = results_df[metric].sort_values(ascending=(metric != 'R2'))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(vals)))
        if metric != 'R2':
            colors = colors[::-1]
        vals.plot.barh(ax=ax, color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} (LOOCV)')
        if metric == 'R2':
            ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.suptitle('Model Comparison -- LOOCV Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = OUTPUT_DIR / 'model_comparison.png'
    plt.savefig(path, dpi=150)
    plt.close()
    plots['model_comparison'] = str(path)
    
    best_name = results_df['R2'].idxmax()
    return best_name, results_df


# ============================================================================
# 6. Main
# ============================================================================
def main():
    print("=" * 60)
    print("  vGFR ML Model -- POC with MLflow Tracking")
    print("=" * 60)
    print()

    # Setup MLflow
    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load
    df, data_info = load_data()

    # Features
    X_raw, y = select_features(df)
    X, engineered_names = engineer_features(X_raw)

    # Feature analysis
    correlations, feat_analysis = analyze_features(X, y)

    # EDA plots
    eda_plots = plot_eda(X, y, correlations, feat_analysis)

    # Save feature analysis as CSV
    feat_analysis_path = OUTPUT_DIR / 'feature_analysis.csv'
    feat_analysis.to_csv(feat_analysis_path, index=False)

    # --- Parent MLflow run ---
    with mlflow.start_run(run_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Log data versioning
        mlflow.log_params({
            'data_hash': data_info['data_hash'],
            'n_patients': data_info['n_rows'],
            'n_raw_features': len(X_raw.columns),
            'n_engineered_features': len(engineered_names),
            'n_total_features': len(X.columns),
            'target': TARGET,
            'excluded_prefixes': str(EXCLUDE_PREFIXES),
        })

        mlflow.log_metrics({
            'target_mean': data_info['target_mean'],
            'target_std': data_info['target_std'],
            'target_min': data_info['target_min'],
            'target_max': data_info['target_max'],
        })

        # Log data info as JSON artifact
        data_info_serializable = {k: v for k, v in data_info.items()}
        data_info_serializable['null_counts'] = {k: int(v) for k, v in data_info['null_counts'].items()}
        data_info_path = OUTPUT_DIR / 'data_info.json'
        with open(data_info_path, 'w') as f:
            json.dump(data_info_serializable, f, indent=2, default=str)
        mlflow.log_artifact(str(data_info_path))

        # Log feature analysis
        mlflow.log_artifact(str(feat_analysis_path))

        # Log raw and engineered feature lists
        feature_lists = {
            'raw_features': list(X_raw.columns),
            'engineered_features': engineered_names,
            'all_features_sorted_by_correlation': correlations.index.tolist(),
        }
        feat_list_path = OUTPUT_DIR / 'feature_lists.json'
        with open(feat_list_path, 'w') as f:
            json.dump(feature_lists, f, indent=2)
        mlflow.log_artifact(str(feat_list_path))

        # SKIP logging raw EDA plots to MLflow to save space
        # for plot_name, plot_path in eda_plots.items():
        #     mlflow.log_artifact(plot_path)

        # Train & evaluate all models (nested runs)
        results, predictions = train_and_evaluate(
            X, y, correlations, feat_analysis, data_info, eda_plots
        )

        # Summary plots
        best_name, results_df = plot_summary(
            results, predictions, y, X, correlations
        )

        champ_metrics = results[best_name]
        champ_y_pred = predictions[best_name]
        k_best = champ_metrics['k']
        sorted_feats = correlations.abs().sort_values(ascending=False).index.tolist()[:k_best]
        model_label = f"R1 Champ - {best_name}"

        # 1. Main Scatter Plot
        plot_path = OUTPUT_DIR / f'round01_champion_{best_name}.png'
        plot_egfrc_vs_vgfr(y.values, champ_y_pred, model_label, sorted_feats, champ_metrics, plot_path)
        mlflow.log_artifact(str(plot_path))

        # 2. Residual Plot
        res_path = OUTPUT_DIR / f'round01_champion_{best_name}_residuals.png'
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

        # Log best model info at parent level
        mlflow.log_params({
            'best_model': best_name,
            'best_model_k': results[best_name]['k'],
        })
        mlflow.log_metrics({
            'best_MAE': results[best_name]['MAE'],
            'best_RMSE': results[best_name]['RMSE'],
            'best_R2': results[best_name]['R2'],
        })

        # Save results table
        results_df.to_csv(OUTPUT_DIR / 'model_results.csv')
        mlflow.log_artifact(str(OUTPUT_DIR / 'model_results.csv'))

    # Console summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset:     {len(y)} patients (hash: {data_info['data_hash']})")
    print(f"  Features:    {len(X_raw.columns)} raw + {len(engineered_names)} engineered = {len(X.columns)} total")
    print(f"  Target:      eGFRc (mean={y.mean():.1f}, std={y.std():.1f})")
    print(f"  Best model:  {best_name}")
    print(f"  Best R2:     {results_df.loc[best_name, 'R2']:.3f}")
    print(f"  Best MAE:    {results_df.loc[best_name, 'MAE']:.2f} mL/min/1.73m2")
    print(f"  Best RMSE:   {results_df.loc[best_name, 'RMSE']:.2f} mL/min/1.73m2")
    print(f"\n  Results:     {OUTPUT_DIR}")
    print(f"  MLflow:      {MLFLOW_DIR}")
    print(f"\n  To view the MLflow UI:")
    print(f"    mlflow ui --backend-store-uri file:///{MLFLOW_DIR.as_posix()} --port 5000")
    print()


if __name__ == '__main__':
    main()
