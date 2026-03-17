# Gold Table Creation - Complete Summary

## ✅ What Was Accomplished

Successfully created `gold.anon_segmentations_with_egfr` table with:
1. ✅ Left join of `bronze.anon_segmentations` with `bronze.anon_egfr`
2. ✅ Matching on `record_id`
3. ✅ Finding closest `egfr_date` to `scan_date`
4. ✅ Calculated `egfrc` column using the CKD-EPI formula

---

## 📊 Table Details

- **Table Name**: `gold.anon_segmentations_with_egfr`
- **Location**: `database/egfr_data.duckdb`
- **Total Records**: 25
- **Total Columns**: 43
- **Records with eGFR data**: 25 (100%)

---

## 🧮 eGFRc Formula

```
eGFRc = 141 × (PCC/(0.9×88.4))^α × 0.993^β
```

**Where:**
- **PCC** = `serum_creatinine` (μmol/L)
- **α** = -0.411 if PCC ≤ 80 μmol/L, else -1.209
- **β** = `current_age` (years)

**Statistics:**
- Mean: 95.61 mL/min/1.73m²
- Min: 47.65 mL/min/1.73m²
- Max: 124.46 mL/min/1.73m²
- Std Dev: 15.79 mL/min/1.73m²

---

## 📁 Files Created

### 1. Main Script
**`scripts/02_create_gold_anon_segmentations_with_egfr.py`**
- Performs the merge
- Calculates eGFRc
- Creates the gold table

**To run:**
```bash
conda activate nrrd-viewer
python scripts/02_create_gold_anon_segmentations_with_egfr.py
```

### 2. Documentation Files
- **`GOLD_TABLE_SUMMARY.md`** - Complete documentation with notebook cells
- **`explore/comparison_egfr_vs_egfrc.md`** - Dedicated comparison cell

---

## 📝 Quick Start Notebook Cells

### Cell 1: Load the data
```python
import duckdb
import pandas as pd
from pathlib import Path

db_path = Path('../database/egfr_data.duckdb')
conn = duckdb.connect(str(db_path), read_only=True)
anon_seg_egfr_df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').df()
conn.close()

print(f"✓ Loaded {len(anon_seg_egfr_df)} records")
print(f"✓ Columns: {len(anon_seg_egfr_df.columns)}")

# Show key columns
anon_seg_egfr_df[['record_id', 'scan_date', 'egfr_date', 'egfr_value', 
                   'egfrc', 'date_diff_days']].head()
```

### Cell 2: Compare eGFR vs eGFRc
See the detailed comparison cell in: **`explore/comparison_egfr_vs_egfrc.md`**

This cell provides:
- Side-by-side comparison table
- Statistics on differences
- Scatter plot with identity line
- Bland-Altman plot for agreement analysis
- Identification of discrepancies

---

## 🔑 Key Columns in Gold Table

### From Segmentations (37 columns):
- `record_id`, `current_age`, `sex`, `scan_date`
- Arterial phase: 15 measurements
- Venous phase: 11 measurements  
- Late phase: 11 measurements

### From eGFR (4 columns):
- `egfr_date` - Closest eGFR measurement date
- `egfr_value` - Reported eGFR (may contain '>90')
- `serum_creatinine` - In μmol/L
- `redcap_repeat_instance` - Instance number

### Calculated (2 columns):
- `egfrc` - Calculated eGFR using CKD-EPI formula
- `date_diff_days` - Days between scan and eGFR measurement

---

## 🎯 Use Cases

### 1. Analyze kidney function with imaging
```python
# Get records with good kidney function
good_function = anon_seg_egfr_df[anon_seg_egfr_df['egfrc'] >= 90]
print(f"Records with eGFRc ≥ 90: {len(good_function)}")
```

### 2. Correlate eGFRc with kidney measurements
```python
corr = anon_seg_egfr_df[['egfrc', 'arterial_left_kidney_artery']].corr()
print(f"Correlation: {corr.iloc[0,1]:.3f}")
```

### 3. Filter by date proximity
```python
# Get records where eGFR was measured within 30 days of scan
close_matches = anon_seg_egfr_df[anon_seg_egfr_df['date_diff_days'] <= 30]
print(f"Records with eGFR within 30 days: {len(close_matches)}")
```

### 4. Compare reported vs calculated eGFR
```python
# See explore/comparison_egfr_vs_egfrc.md for complete cell
comparison = anon_seg_egfr_df[['record_id', 'egfr_value', 'egfrc']]
```

---

## 🔄 Updating the Table

If new data is added to bronze tables, simply re-run:
```bash
python scripts/02_create_gold_anon_segmentations_with_egfr.py
```

The script will:
1. Drop the existing gold table
2. Re-merge the data
3. Recalculate eGFRc
4. Create the updated table

---

## 📚 Additional Resources

- **Full notebook examples**: `GOLD_TABLE_SUMMARY.md`
- **Comparison analysis**: `explore/comparison_egfr_vs_egfrc.md`
- **Original notebook**: `explore/case_explorer.ipynb`

---

## ✨ Next Steps

1. Open `case_explorer.ipynb` in Jupyter
2. Add cells from the documentation files
3. Run the analysis
4. Explore correlations between eGFRc and kidney measurements
5. Compare reported eGFR with calculated eGFRc

**Happy analyzing! 🚀**
