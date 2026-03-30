import sys
from pathlib import Path

# Add project root to path
sys.path.append(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline')

import duckdb
import pandas as pd
from ml_utils import load_cohort, get_feature_matrix

def main():
    df, _ = load_cohort('12-03-2026')
    X, _ = get_feature_matrix(df, exclude_vol_hu=False)
    
    print("\n--- ALL FEATURES ---")
    features = sorted(X.columns)
    
    # Simple keyword categorizer
    for feat in features:
        if 'art_flow_efficiency' in feat or 'ven_flow_efficiency' in feat or 'mean_artery_venous' in feat or 'mean_artery_arterial' in feat:
            phase = "DERIVED (Cross-Phase/Composite)"
        elif 'arterial' in feat:
            phase = "ARTERIAL"
        elif 'venous' in feat:
            phase = "VENOUS"
        elif 'late' in feat or 'E_late' in feat:
            phase = "LATE"
        else:
            phase = "GLOBAL / DEMOGRAPHIC"
            
        print(f"[{phase:28}] {feat}")

if __name__ == "__main__":
    main()
