"""
Plot eGFRc vs vGFR (Predicted) for any model configuration.

This script loads data, runs LOOCV for a specified model, and generates
a publication-quality scatter plot of actual eGFRc vs predicted vGFR.

Usage:
    python plot_egfrc_vs_vgfr.py                          # uses best known config (stepwise)
    python plot_egfrc_vs_vgfr.py --model ridge --k 10 --alpha 10
    python plot_egfrc_vs_vgfr.py --model ridge --features stepwise
    python plot_egfrc_vs_vgfr.py --model bayesian_ridge --k 10
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data.duckdb'
OUTPUT_DIR = Path(__file__).parent / 'ml_results'
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = 'egfrc'
EXCLUDE_PREFIXES = ['vgfr_', 'conc_lit_', 'conc_late_', 'w_pv_', 'w_back_']
EXCLUDE_COLS = [
    'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
    'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days',
    TARGET,
]

# Best known feature set from forward stepwise selection (EXP2)
STEPWISE_FEATURES = [
    'current_age',
    'arterial_venecava_between_kidney_hepatic',
    'E_late_right',
    'norm_vein_arterial_right',
    'arterial_portal_vein',
    'LR_ratio_vein_venous',
    'mean_artery_venous',
    'late_venacava_above_hepatic',
    'late_left_hepatic_vein',
    'E_arterial_right',
    'norm_artery_venous_right',
]


# ============================================================================
# Data loading & feature engineering (reused from vgfr_improvements.py)
# ============================================================================
def load_and_prepare():
    """Load data, select features, engineer features."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').fetchdf()
    conn.close()

    # Select features
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

    # Engineer features
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
    return X, y


def get_top_k(X, y, k):
    """Top-k features by absolute correlation with target."""
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    return corrs.index[:k].tolist()


def build_estimator(model_name, alpha=10.0):
    """Build a sklearn pipeline from model name."""
    models = {
        'ridge': Ridge(alpha=alpha),
        'lasso': Lasso(alpha=alpha, max_iter=10000),
        'elasticnet': ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000),
        'bayesian_ridge': BayesianRidge(),
        'huber': HuberRegressor(epsilon=1.35, alpha=alpha),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        'gbr': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
    }
    model = models.get(model_name.lower())
    if model is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")

    if model_name.lower() in ('rf', 'gbr'):
        return Pipeline([('model', model)])
    else:
        return Pipeline([('scaler', StandardScaler()), ('model', model)])


# ============================================================================
# Plot
# ============================================================================
def plot_egfrc_vs_vgfr(y_actual, y_predicted, model_label, features_used, metrics, output_path):
    """
    Generate publication-quality eGFRc vs vGFR (Predicted) scatter plot.

    Parameters:
        y_actual: actual eGFRc values (from creatinine formula)
        y_predicted: model-predicted vGFR values (from CT imaging)
        model_label: model description string
        features_used: list of feature names
        metrics: dict with R2, MAE, RMSE
        output_path: where to save the plot
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    # Scatter
    sc = ax.scatter(y_predicted, y_actual, s=120, alpha=0.8,
                    edgecolors='black', linewidth=0.8, c='steelblue', zorder=5)

    # Identity line
    margin = 10
    lims = [min(y_actual.min(), y_predicted.min()) - margin,
            max(y_actual.max(), y_predicted.max()) + margin]
    ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7, label='Identity (perfect prediction)', zorder=3)

    # +/- MAE band
    mae = metrics['MAE']
    ax.fill_between(lims, [l - mae for l in lims], [l + mae for l in lims],
                    alpha=0.08, color='red', label=f'+/- MAE ({mae:.1f})', zorder=2)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('vGFR (Predicted from CT) [mL/min/1.73m$^2$]', fontsize=13)
    ax.set_ylabel('eGFRc (Actual - Creatinine) [mL/min/1.73m$^2$]', fontsize=13)
    ax.set_title('eGFRc vs vGFR (Predicted from CT Imaging)', fontsize=15, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    # Metrics annotation box
    textstr = (f'Model: {model_label}\n'
               f'R$^2$ = {metrics["R2"]:.3f}\n'
               f'MAE = {metrics["MAE"]:.2f} mL/min/1.73m$^2$\n'
               f'RMSE = {metrics["RMSE"]:.2f} mL/min/1.73m$^2$\n'
               f'n = {len(y_actual)} patients\n'
               f'Features: {len(features_used)}')
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Plot eGFRc vs vGFR for any model')
    parser.add_argument('--model', type=str, default='ridge',
                        help='Model type: ridge, lasso, elasticnet, bayesian_ridge, huber, rf, gbr')
    parser.add_argument('--features', type=str, default='stepwise',
                        help='Feature selection: "stepwise" (best known), "corr" (correlation-based)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of features when using --features corr (default: 10)')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Regularization strength (default: 10.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    args = parser.parse_args()

    # Load data
    X, y = load_and_prepare()
    y_arr = y.values

    # Select features
    if args.features == 'stepwise':
        features = [f for f in STEPWISE_FEATURES if f in X.columns]
        feat_label = 'stepwise'
    elif args.features == 'corr':
        features = get_top_k(X, y, args.k)
        feat_label = f'corr_top{args.k}'
    else:
        raise ValueError(f"Unknown feature selection: {args.features}")

    print(f"Model:    {args.model} (alpha={args.alpha})")
    print(f"Features: {feat_label} ({len(features)} features)")
    print(f"  {features}")

    # Build model and run LOOCV
    estimator = build_estimator(args.model, args.alpha)
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X[features].values, y_arr, cv=loo)

    # Metrics
    metrics = {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred),
    }

    print(f"\nResults (LOOCV):")
    print(f"  R2   = {metrics['R2']:.3f}")
    print(f"  MAE  = {metrics['MAE']:.2f}")
    print(f"  RMSE = {metrics['RMSE']:.2f}")

    # Model label
    model_label = f"{args.model.title()} (a={args.alpha}, {feat_label})"

    # Output path
    if args.output:
        out_path = OUTPUT_DIR / args.output
    else:
        out_path = OUTPUT_DIR / f'egfrc_vs_vgfr_{args.model}_{feat_label}.png'

    # Plot
    plot_egfrc_vs_vgfr(y_arr, y_pred, model_label, features, metrics, out_path)


if __name__ == '__main__':
    main()
