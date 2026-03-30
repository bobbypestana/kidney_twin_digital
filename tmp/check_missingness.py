import sys
sys.path.append(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline')
import duckdb
import pandas as pd
from ml_utils import load_cohort

df, _ = load_cohort('12-03-2026')

# Exclude global/metadata
meta = ['record_id', 'sex', 'scan_date', 'egfr_date', 'egfr_value', 'serum_creatinine', 'redcap_repeat_instance', 'date_diff_days', 'source_folder', 'egfrc']
features = [c for c in df.columns if c not in meta]

total = len(df)
print(f"Total Patients in 12-03-2026: {total}\n")

missing_rates = []
for f in features:
    missing_count = df[f].isna().sum()
    rate = missing_count / total
    missing_rates.append((f, missing_count, rate))

missing_rates.sort(key=lambda x: x[2], reverse=True)

print("Top 20 Features with HIGHEST missing rates:")
for f, count, rate in missing_rates[:20]:
    print(f"{rate*100:5.1f}% ({count:2d}/{total}) - {f}")

print("\nMissing rates for Champion GradientBoosting features:")
champ_feats = [
    'norm_ven_artery_right', 'venous_left_kidney_vein', 'current_age',
    'venous_right_kidney_vein', 'venous_venacava_between_hu_std',
    'venous_venacava_above_hu_mean', 'venous_venacava_above_hu_std'
]
for f in champ_feats:
    if f in df.columns:
        count = df[f].isna().sum()
        rate = count / total
        print(f"{rate*100:5.1f}% ({count:2d}/{total}) - {f}")
    else:
        print(f"Feature {f} not in df (might be engineered)")
