import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pathlib

# ============================================================================
# 1. Feature Engineering Logic (Mirroring gold_layer.py)
# ============================================================================

def get_new_cohort_features(conn):
    # This imitates the gold_layer logic but only in-memory
    query_master = """
    WITH ranked_egfr AS (
        SELECT 
            s.* EXCLUDE (current_age, sex),
            COALESCE(e.current_age, s.current_age) AS current_age,
            COALESCE(e.sex, s.sex) AS sex,
            e.egfr_date,
            e.egfr_value,
            e.serum_creatinine,
            e.egfrc,
            ABS(DATEDIFF('day', TRY_CAST(s.scan_date AS DATE), TRY_CAST(e.egfr_date AS DATE))) AS date_diff_days,
            ROW_NUMBER() OVER (
                PARTITION BY s.record_id, s.scan_date 
                ORDER BY ABS(DATEDIFF('day', TRY_CAST(s.scan_date AS DATE), TRY_CAST(e.egfr_date AS DATE))) ASC
            ) as rn
        FROM silver.segmentations s
        LEFT JOIN silver.egfr e ON s.record_id = e.record_id
        WHERE s.source_folder = '12-03-2026'
    )
    SELECT * EXCLUDE (rn) FROM ranked_egfr WHERE rn = 1
    """
    master_cases = conn.execute(query_master).df()
    
    df = master_cases.copy()
    phases = ['arterial', 'venous', 'late']
    for p in phases:
        art_col = f"{p}_left_kidney_artery"
        vein_col = f"{p}_left_kidney_vein"
        if art_col in df.columns and vein_col in df.columns:
            df[f'E_{p}_left'] = (df[art_col] - df[vein_col]) / df[art_col].replace(0, np.nan)
            
        art_col = f"{p}_right_kidney_artery"
        vein_col = f"{p}_right_kidney_vein"
        if art_col in df.columns and vein_col in df.columns:
            df[f'E_{p}_right'] = (df[art_col] - df[vein_col]) / df[art_col].replace(0, np.nan)
    
    if 'arterial_aorta' in df.columns:
        df['norm_art_artery_left'] = df['arterial_left_kidney_artery'] / df['arterial_aorta'].replace(0, np.nan)
        df['norm_art_artery_right'] = df['arterial_right_kidney_artery'] / df['arterial_aorta'].replace(0, np.nan)
    
    for p in phases:
        left = f'E_{p}_left'
        right = f'E_{p}_right'
        if left in df.columns and right in df.columns:
            df[f'E_{p}_mean'] = (df[left] + df[right]) / 2
            
    df['mean_artery_arterial'] = (df['arterial_left_kidney_artery'] + df['arterial_right_kidney_artery']) / 2
    df['mean_artery_venous'] = (df['venous_left_kidney_artery'] + df['venous_right_kidney_artery']) / 2
    df['age_x_E_arterial'] = df['current_age'] * df.get('E_arterial_mean', 0)
    df['art_flow_efficiency'] = df.get('E_arterial_mean', 0) * df['mean_artery_arterial']
    df['ven_flow_efficiency'] = df.get('E_venous_mean', 0) * df['mean_artery_venous']
    
    return df

# ============================================================================
# 2. Main Execution
# ============================================================================

FEATURES = ['current_age', 'E_late_right', 'venous_kidney_vol', 'venous_venacava_above_hepatic', 
            'arterial_venecava_between_kidney_hepatic', 'E_late_mean', 'arterial_kidney_hu_std', 
            'E_arterial_left', 'venous_right_kidney_artery', 'venous_right_hepatic_vein', 
            'venous_kidney_hu_median', 'late_kidney_hu_std']

def main():
    # Use relative path to database if script is in root
    db_path = 'database/egfr_data_v2.duckdb'
    conn = duckdb.connect(db_path)
    
    train_df = conn.execute("SELECT * FROM gold.ml_features WHERE source_folder = '25-11-2025'").df()
    pred_df = get_new_cohort_features(conn)
    X_train = train_df[FEATURES]
    y_train = train_df['egfrc']
    
    for f in FEATURES:
        if f not in pred_df.columns:
            pred_df[f] = np.nan
    X_pred = pred_df[FEATURES]
    
    estimators = [
        ('bayesian', BayesianRidge()),
        ('huber', HuberRegressor()),
        ('svr', SVR(kernel='rbf', C=10.0))
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()), 
        ('model', stack)
    ])
    
    print("Training best model (Round 12) on 255-patient dataset...")
    pipe.fit(X_train, y_train)
    
    print(f"Predicting vGFR for {len(X_pred)} records from 12-03-2026...")
    vgfr_predictions = pipe.predict(X_pred)
    pred_df['predicted_vgfr'] = vgfr_predictions
    
    pred_df['is_discrepant'] = (pd.to_numeric(pred_df['egfr_value'], errors='coerce') - pred_df['egfrc']).abs() > 1
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=pred_df, x='predicted_vgfr', y='egfrc', hue='is_discrepant', style='is_discrepant', 
                    palette={True: 'red', False: 'blue'}, s=100, markers={True: 'X', False: 'o'})
    
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, '--r', alpha=0.5, label='Identity (Pred=eGFRc)')
    
    plt.title("vGFR Predictions for 12-03-2026 Cohort (Best Model: R12 Stacking)")
    plt.xlabel("Predicted vGFR")
    plt.ylabel("Ground Truth (eGFRc)")
    plt.legend(title="eGFR vs eGFRc Discrepancy (>1)")
    plt.grid(True, alpha=0.3)
    
    output_path = '02_ml_pipeline/12032026_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    conn.close()

if __name__ == "__main__":
    main()
