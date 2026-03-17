# gold.segmentations_with_egfr - Summary

## ✅ Table Created Successfully

**Table**: `gold.segmentations_with_egfr`  
**Records**: 25  
**Columns**: 25

---

## 📊 What's Included

### Original Segmentation Data (13 columns):
- `case_number`
- Arterial phase: `arterial_left_kidney_artery`, `arterial_left_kidney_vein`, `arterial_right_kidney_artery`, `arterial_right_kidney_vein`
- Venous phase: `venous_left_kidney_artery`, `venous_left_kidney_vein`, `venous_right_kidney_artery`, `venous_right_kidney_vein`
- Late phase: `late_left_kidney_artery`, `late_left_kidney_vein`, `late_right_kidney_artery`, `late_right_kidney_vein`

### From Anonymized Data (3 columns):
- `egfrc` - Calculated eGFR (matched from `anon_segmentations_with_egfr`)
- `serum_creatinine` - In μmol/L
- `current_age` - Patient age in years

### Calculated vGFR - HU Method Only (9 columns):
**Per kidney:**
- `vgfr_arterial_left`, `vgfr_arterial_right`
- `vgfr_venous_left`, `vgfr_venous_right`
- `vgfr_late_left`, `vgfr_late_right`

**Mean per phase:**
- `vgfr_arterial_mean`
- `vgfr_venous_mean`
- `vgfr_late_mean`

---

## 🧮 Calculations Performed

### eGFRc
- **Source**: Matched from `gold.anon_segmentations_with_egfr` using `record_id = case_number`
- **Formula**: CKD-EPI equation
- **Mean**: 95.61 mL/min/1.73m²
- **Range**: [47.65, 124.46]

### vGFR (HU Method, No Corrections)
```
vGFR = RPF × (P_RA - P_RV) / P_RA

Where:
- RPF = 600 mL/min (fixed)
- P_RA = HU in renal artery
- P_RV = HU in renal vein
```

**Statistics:**
- **Arterial**: Mean = 153.46 mL/min, Range = [8.79, 328.28]
- **Venous**: Mean = -26.35 mL/min, Range = [-148.22, 47.57]
- **Late**: Mean = -23.95 mL/min, Range = [-124.15, 51.35]

---

## 📝 Quick Start

### Load the table:
```python
import duckdb
from pathlib import Path

db_path = Path('../database/egfr_data.duckdb')
conn = duckdb.connect(str(db_path), read_only=True)
seg_egfr_df = conn.execute('SELECT * FROM gold.segmentations_with_egfr').df()
conn.close()

print(f"Loaded {len(seg_egfr_df)} records with {len(seg_egfr_df.columns)} columns")
```

### View key columns:
```python
key_cols = ['case_number', 'egfrc', 'vgfr_arterial_mean', 
            'vgfr_venous_mean', 'vgfr_late_mean']
seg_egfr_df[key_cols].head(10)
```

### Compare vGFR with eGFRc:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(seg_egfr_df['vgfr_arterial_mean'], seg_egfr_df['egfrc'], alpha=0.6, s=100)
plt.xlabel('vGFR Arterial (mL/min)', fontsize=12)
plt.ylabel('eGFRc (mL/min/1.73m²)', fontsize=12)
plt.title('vGFR vs eGFRc', fontsize=14)
plt.grid(True, alpha=0.3)

# Calculate correlation
corr = seg_egfr_df[['vgfr_arterial_mean', 'egfrc']].corr().iloc[0, 1]
plt.text(0.05, 0.95, f'r = {corr:.3f}', 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()
```

---

## 🎯 Analysis Ready

This table is now ready for the **first analysis**: HU units with no correction for all phases.

You can:
1. ✅ Compare vGFR across phases (arterial, venous, late)
2. ✅ Correlate vGFR with eGFRc
3. ✅ Analyze left vs right kidney function
4. ✅ Identify patterns and outliers

---

## 🔄 Recreate Table

To recreate or update the table:

```bash
conda activate nrrd-viewer
python scripts/03_create_gold_segmentations_with_egfr.py
```

---

## 📋 Comparison with Anonymized Table

Both tables now have the same structure:
- `gold.anon_segmentations_with_egfr` - 99 columns (includes all conversion methods and corrections)
- `gold.segmentations_with_egfr` - 25 columns (basic: eGFRc + vGFR HU method only)

**Ready for analysis!** 🚀
