import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path('database/egfr_data.duckdb')
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent.parent / 'database' / 'egfr_data.duckdb'

OUTPUT_DIR = Path('classical_ml/ml_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'egfrc'

EXCLUDE_PREFIXES = ['vgfr_', 'conc_lit_', 'conc_late_', 'w_pv_', 'w_back_']
EXCLUDE_COLS = [
    'record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value',
    'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days', TARGET,
]

def load_and_prepare():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').fetchdf()
    conn.close()

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

    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median()), y

def evaluate_loocv(X_arr, y_arr, estimator):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X_arr, y_arr, cv=loo, n_jobs=-1)
    return y_pred, {
        'MAE': mean_absolute_error(y_arr, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_arr, y_pred)),
        'R2': r2_score(y_arr, y_pred)
    }

def main():
    X, y = load_and_prepare()
    
    champions = {
        'FINAL_Best_R2_RMSE_BayesianRidge': {
            'model': BayesianRidge(),
            'features': [
                'current_age', 'arterial_venecava_between_kidney_hepatic',
                'E_late_right', 'norm_vein_arterial_right', 'arterial_portal_vein',
                'LR_ratio_vein_venous', 'mean_artery_venous',
                'late_venacava_above_hepatic', 'late_left_hepatic_vein',
                'E_arterial_right', 'norm_artery_venous_right'
            ],
            'color': 'steelblue',
            'title': 'Highest R² and Lowest RMSE'
        },
        'FINAL_Best_MAE_SVR': {
            'model': SVR(kernel='rbf', C=10.0, gamma='scale'),
            'features': [
                'current_age', 'arterial_venacava_below_kidney',
                'arterial_right_kidney_vein', 'norm_vein_late_right'
            ],
            'color': 'indigo',
            'title': 'Highest Precision (Lowest MAE)'
        },
        'FINAL_Best_Balanced_Ridge': {
            'model': Ridge(alpha=10.0),
            'features': [
                "current_age", "arterial_venecava_between_kidney_hepatic",
                "E_late_right", "E_arterial_right", "arterial_portal_vein",
                "norm_artery_venous_right", "LR_ratio_vein_venous"
            ],
            'color': 'forestgreen',
            'title': 'Best Balanced Trade-off'
        }
    }
    
    print("Evaluating Final Champions...")
    
    for name, config in champions.items():
        feats = config['features']
        est = Pipeline([('scaler', StandardScaler()), ('model', config['model'])])
        
        y_pred, metrics = evaluate_loocv(X[feats].values, y.values, est)
        
        print(f"\n{name} ({len(feats)} features)")
        print(f"  MAE:  {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R2:   {metrics['R2']:.3f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_pred, y.values, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, c=config['color'])
        lims = [min(y.min(), y_pred.min()) - 5, max(y.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7)
        
        mae_val = metrics['MAE']
        ax.fill_between(lims, [l - mae_val for l in lims], [l + mae_val for l in lims],
                        alpha=0.08, color='red', label=f'+/- MAE ({mae_val:.1f})')
                        
        ax.set_xlim(lims); ax.set_ylim(lims); 
        ax.set_xlabel('Predicted vGFR (CT Imaging)')
        ax.set_ylabel('Actual eGFRc (Creatinine-based)')
        
        ax.set_title(f"CHAMPION: {config['title']}\n"
                     f"{config['model'].__class__.__name__} | {len(feats)} features\n"
                     f"R²={metrics['R2']:.3f} | MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f}",
                     pad=15)
                     
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        plt.tight_layout()
        
        p = OUTPUT_DIR / f'{name}.png'
        plt.savefig(p, dpi=200)
        plt.close()

if __name__ == '__main__':
    main()
