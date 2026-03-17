import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = Path(r'g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb')
TARGET = 'egfrc'

def base_engineer(X):
    X = X.copy()
    for phase in ['arterial', 'venous', 'late']:
        for side in ['left', 'right']:
            art = f'{phase}_{side}_kidney_artery'
            vein = f'{phase}_{side}_kidney_vein'
            if art in X.columns and vein in X.columns:
                X[f'E_{phase}_{side}'] = (X[art] - X[vein]) / X[art].replace(0, np.nan)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    return X

def main():
    print("=" * 60)
    print("  Sanity Check ML Script (v2 Database)")
    print("=" * 60)

    # 1. Load Data
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.master_cases').df()
    conn.close()
    
    print(f"[OK] Loaded {len(df)} patients from gold.master_cases")

    # 2. Extract Features (Simulate vgfr_improvements logic)
    # The original script drops metadata and keeps numeric HU columns
    drop_cols = [
        'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
        'serum_creatinine', 'date_diff_days', 'source_folder', TARGET
    ]
    # Also drop columns starting with certain prefixes
    for col in df.columns:
        if any(col.startswith(p) for p in ['vgfr_', 'conc_lit_', 'w_pv_']):
            drop_cols.append(col)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_raw = df[feature_cols].select_dtypes(include=[np.number])
    y = df[TARGET]
    
    print(f"[OK] Selected {X_raw.shape[1]} raw features")

    # 3. Feature Engineering
    X = base_engineer(X_raw)
    
    # 4. EXP 0: Baseline (Ridge top-10, alpha=10)
    # Get top 10 features by correlation
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    feats = corrs.index[:10].tolist()
    print(f"[OK] Selected Top 10 Features: {feats}")

    # 5. Evaluate (LOOCV)
    estimator = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X[feats].values, y.values, cv=loo)
    
    mae = mean_absolute_error(y.values, y_pred)
    r2 = r2_score(y.values, y_pred)
    
    print("\n" + "=" * 60)
    print("RESULTS (LOOCV)")
    print("=" * 60)
    print(f"R2:  {r2:.3f} (Legacy Baseline ~0.406)")
    print(f"MAE: {mae:.2f} (Legacy Baseline ~8.67)")
    print("=" * 60)

if __name__ == '__main__':
    main()
