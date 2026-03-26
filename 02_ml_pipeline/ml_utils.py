"""
ml_utils.py — Shared utilities for the vGFR ML pipeline.
Handles cohort loading, feature matrix prep, and output path naming.
"""

import argparse
import hashlib
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data_v2.duckdb'
OUTPUT_DIR = Path(__file__).parent / 'ml_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'egfrc'

# Columns that are never features
_META_COLS = [
    'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
    'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days',
    'source_folder', TARGET
]

# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(description="vGFR ML experiment"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--cohort',
        type=str,
        default=None,
        help='Filter to a specific cohort source_folder value (e.g. "25-11-2025"). '
             'Omit to use all cohorts combined.'
    )
    parser.add_argument(
        '--exclude-vol-hu',
        action='store_true',
        default=False,
        help='Exclude volume and HU distribution features '
             '(use for cohorts without volume segmentation).'
    )
    return parser.parse_args()


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_cohort(cohort=None):
    """
    Load gold.ml_features, optionally filtering to a single cohort.

    Parameters
    ----------
    cohort : str or None
        source_folder value to filter on, e.g. ``'25-11-2025'``.
        ``None`` loads all cohorts combined.

    Returns
    -------
    df : pd.DataFrame
    data_hash : str (16-char hex)
    """
    # Note: source_folder is excluded from the gold table so we join master_cases
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    if cohort:
        # gold.ml_features no longer has source_folder, so we join master_cases
        df = conn.execute(
            """
            SELECT f.*
            FROM gold.ml_features f
            JOIN gold.master_cases m ON f.record_id = m.record_id
            WHERE m.source_folder = ?
            """,
            [cohort]
        ).fetchdf()
    else:
        df = conn.execute('SELECT * FROM gold.ml_features').fetchdf()
    conn.close()

    data_hash = (
        hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes())
        .hexdigest()[:16]
    )
    return df, data_hash


# ─── Feature Matrix ───────────────────────────────────────────────────────────
def get_feature_matrix(df, exclude_vol_hu=False):
    """
    Build X and y from a gold feature DataFrame.

    - Drops all meta/target columns.
    - Keeps only numeric columns.
    - Optionally drops volume and HU columns (all-NULL on new cohort).
    - Auto-skips columns that are entirely NULL for the given data.
    - Fills remaining NAs with column median.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[c for c in _META_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    if exclude_vol_hu:
        vol_hu_patterns = ['_vol', '_hu_']
        X = X.drop(
            columns=[c for c in X.columns
                     if any(p in c.lower() for p in vol_hu_patterns)],
            errors='ignore'
        )

    # Auto-skip fully-NULL columns (feature doesn't exist for this cohort)
    all_null = X.columns[X.isna().all()]
    if len(all_null):
        print(f"  [auto-skip] {len(all_null)} all-NULL columns dropped: "
              f"{list(all_null)}")
        X = X.drop(columns=all_null)

    X = X.astype(float).fillna(X.median()).fillna(0)
    y = df[TARGET]
    return X, y


# ─── Naming Helpers ───────────────────────────────────────────────────────────
def safe_cohort_tag(cohort=None):
    """
    Convert a cohort string to a filesystem-safe tag.
    '25-11-2025' → '25112025', None → 'all'
    """
    if cohort is None:
        return 'all'
    return re.sub(r'[^A-Za-z0-9]', '', cohort)


def make_output_path(round_name, cohort=None, output_dir=None):
    """
    Build a plot path: ``{output_dir}/{cohort_tag}_{round_name}_champion.png``

    Examples
    --------
    make_output_path('round_12', '25-11-2025')
      → ml_results/25112025_round_12_champion.png
    make_output_path('round_12')
      → ml_results/all_round_12_champion.png
    """
    base = output_dir or OUTPUT_DIR
    tag = safe_cohort_tag(cohort)
    return Path(base) / f"{tag}_{round_name}_champion.png"


def experiment_name(base_name, cohort=None):
    """
    Append cohort tag to an MLflow experiment name.
    e.g. 'vGFR_Repro_R2_R10' + '25-11-2025' → 'vGFR_Repro_R2_R10_25112025'
    """
    tag = safe_cohort_tag(cohort)
    return f"{base_name}_{tag}"


# ─── Summary Banner ───────────────────────────────────────────────────────────
# ─── Summary Banner ───────────────────────────────────────────────────────────
def print_run_banner(script_name, cohort, df, X):
    tag = safe_cohort_tag(cohort)
    print("=" * 60)
    print(f"  {script_name}")
    print(f"  Cohort   : {cohort or 'ALL'} (tag={tag})")
    print(f"  Records  : {len(df)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Target   : {TARGET}")
    print("=" * 60)

# ─── Export Utilities ────────────────────────────────────────────────────────
def export_champion_details(cohort_id, round_name, model_name, features, estimator, metrics, y_true, y_pred, story="", is_report_champion=False):
    """
    Save all clinical artifacts (Weights, Residuals, Distributions) for a champion model.
    Organized in ml_results/{cohort_tag}/
    If is_report_champion=True, also copies main plots to results/figures/ for LaTeX.
    """
    import json
    import shutil
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    tag = safe_cohort_tag(cohort_id)
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Report freeze directory
    report_fig_dir = Path(__file__).parent.parent / 'results' / 'figures'
    report_fig_dir.mkdir(parents=True, exist_ok=True)

    # Standardize round name to match tag (e.g. 'Round 3' -> 'round_3')
    round_tag = round_name.lower().replace(' ', '_')
    
    print(f"  [export] Saving {round_name} champion details to {out_dir}...")

    # 1. Weights CSV (if linear)
    if hasattr(estimator, 'named_steps'): # It's a pipeline
        model = estimator.named_steps['model']
    else:
        model = estimator

    if hasattr(model, 'coef_'):
        weights = model.coef_
        pd.DataFrame({'Feature': features, 'Weight': weights}).to_csv(out_dir / f"{round_tag}_weights.csv", index=False)
    
    # 2. Relevance Plot
    if hasattr(model, 'coef_'):
        df_p = pd.DataFrame({'Feature': features, 'Weight': model.coef_}).sort_values('Weight', key=abs, ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_p, x='Weight', y='Feature', palette='vlag', orient='h')
        plt.title(f"Feature Relevance: {round_name}")
        plt.tight_layout()
        relevance_path = out_dir / f"{round_tag}_relevance.png"
        plt.savefig(relevance_path)
        plt.close()
        
        if is_report_champion:
            shutil.copy2(relevance_path, report_fig_dir / f"{round_tag}_feature_relevance.png")

    # 3. Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted vGFR')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f"Residual Analysis: {round_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"{round_tag}_residuals.png")
    plt.close()

    # 4. Distribution (Violin)
    df_dist = pd.DataFrame({
        'Source': ['Actual']*len(y_true) + ['Predicted']*len(y_pred),
        'vGFR': np.concatenate([y_true, y_pred])
    })
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_dist, x='Source', y='vGFR', inner='quart')
    plt.title(f"Prediction Distribution: {round_name}")
    plt.savefig(out_dir / f"{round_tag}_distribution.png")
    plt.close()

    # 5. Champion Plot (Identity) - Copy to Report if needed
    if is_report_champion:
        tag_short = safe_cohort_tag(cohort_id)
        src_plot = OUTPUT_DIR / f"{tag_short}_{round_tag}_champion.png"
        if src_plot.exists():
            shutil.copy2(src_plot, report_fig_dir / f"{round_tag}_champion_plot_official.png")

    # 6. Summary JSON
    summary = {
        'cohort': cohort_id,
        'round': round_name,
        'model': model_name,
        'metrics': metrics,
        'features': features,
        'clinical_story': story
    }
    with open(out_dir / f"{round_tag}_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
