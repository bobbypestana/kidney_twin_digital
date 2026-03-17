# vGFR Calculation Summary

## ✅ What Was Added

Successfully added **volumetric GFR (vGFR)** calculations to the gold table.

---

## 📊 New Columns Added (9 total)

### Per Kidney and Phase (6 columns):
1. `vgfr_arterial_left` - Left kidney, arterial phase
2. `vgfr_arterial_right` - Right kidney, arterial phase
3. `vgfr_venous_left` - Left kidney, venous phase
4. `vgfr_venous_right` - Right kidney, venous phase
5. `vgfr_late_left` - Left kidney, late phase
6. `vgfr_late_right` - Right kidney, late phase

### Mean Per Phase (3 columns):
7. `vgfr_arterial_mean` - Average of left and right, arterial phase
8. `vgfr_venous_mean` - Average of left and right, venous phase
9. `vgfr_late_mean` - Average of left and right, late phase

---

## 🧮 vGFR Formula

```
vGFR = RPF × E

where:
E = (P_RA - P_RV) / P_RA  (Extraction Ratio)

Therefore:
vGFR = RPF × (P_RA - P_RV) / P_RA
```

**Parameters:**
- **RPF** = Renal Plasma Flow = **600 mL/min** (fixed, literature-based)
- **P_RA** = Iodine concentration in renal artery (from CT scan)
- **P_RV** = Iodine concentration in renal vein (from CT scan)
- **E** = Extraction ratio (fraction of iodine extracted in one pass)

**Units:** mL/min

---

## 📋 Column Mapping

### Arterial Phase:
- P_RA (left) = `arterial_left_kidney_artery`
- P_RV (left) = `arterial_left_kidney_vein`
- P_RA (right) = `arterial_right_kidney_artery`
- P_RV (right) = `arterial_right_kidney_vein`

### Venous Phase:
- P_RA (left) = `venous_left_kidney_artery`
- P_RV (left) = `venous_left_kidney_vein`
- P_RA (right) = `venous_right_kidney_artery`
- P_RV (right) = `venous_right_kidney_vein`

### Late Phase:
- P_RA (left) = `late_left_kidney_artery`
- P_RV (left) = `late_left_kidney_vein`
- P_RA (right) = `late_right_kidney_artery`
- P_RV (right) = `late_right_kidney_vein`

---

## 📊 Updated Table Info

- **Table Name**: `gold.anon_segmentations_with_egfr`
- **Total Records**: 25
- **Total Columns**: 52 (was 43, added 9 vGFR columns)

**Column breakdown:**
- Original segmentation data: 37 columns
- eGFR data: 4 columns
- Calculated eGFRc: 1 column
- Date difference: 1 column
- **vGFR calculations: 9 columns** ✨

---

## 📝 Notebook Cell Examples

### Cell 1: Load and view vGFR data
```python
import duckdb
import pandas as pd
from pathlib import Path

# Load data
db_path = Path('../database/egfr_data.duckdb')
conn = duckdb.connect(str(db_path), read_only=True)
df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').df()
conn.close()

# Display vGFR columns
vgfr_cols = ['record_id', 'vgfr_arterial_left', 'vgfr_arterial_right', 'vgfr_arterial_mean',
             'vgfr_venous_left', 'vgfr_venous_right', 'vgfr_venous_mean',
             'vgfr_late_left', 'vgfr_late_right', 'vgfr_late_mean']

print("vGFR Data (first 5 records):")
print("="*80)
df[vgfr_cols].head()
```

### Cell 2: vGFR Statistics by Phase
```python
# Calculate statistics for each phase
print("vGFR STATISTICS BY PHASE")
print("="*80)

phases = ['arterial', 'venous', 'late']

for phase in phases:
    print(f"\n{phase.upper()} PHASE:")
    print("-"*80)
    
    left_col = f'vgfr_{phase}_left'
    right_col = f'vgfr_{phase}_right'
    mean_col = f'vgfr_{phase}_mean'
    
    print(f"Left Kidney:")
    print(f"  Mean: {df[left_col].mean():.2f} mL/min")
    print(f"  Median: {df[left_col].median():.2f} mL/min")
    print(f"  Range: [{df[left_col].min():.2f}, {df[left_col].max():.2f}]")
    
    print(f"\nRight Kidney:")
    print(f"  Mean: {df[right_col].mean():.2f} mL/min")
    print(f"  Median: {df[right_col].median():.2f} mL/min")
    print(f"  Range: [{df[right_col].min():.2f}, {df[right_col].max():.2f}]")
    
    print(f"\nBoth Kidneys (Mean):")
    print(f"  Mean: {df[mean_col].mean():.2f} mL/min")
    print(f"  Median: {df[mean_col].median():.2f} mL/min")
    print(f"  Range: [{df[mean_col].min():.2f}, {df[mean_col].max():.2f}]")
```

### Cell 3: Compare vGFR with eGFRc
```python
import matplotlib.pyplot as plt

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

phases = ['arterial', 'venous', 'late']
colors = ['steelblue', 'orange', 'green']

for idx, (phase, color) in enumerate(zip(phases, colors)):
    mean_col = f'vgfr_{phase}_mean'
    
    axes[idx].scatter(df['egfrc'], df[mean_col], alpha=0.6, s=100, color=color)
    axes[idx].set_xlabel('eGFRc (mL/min/1.73m²)', fontsize=12)
    axes[idx].set_ylabel(f'vGFR {phase.title()} (mL/min)', fontsize=12)
    axes[idx].set_title(f'eGFRc vs vGFR ({phase.title()} Phase)', fontsize=14)
    axes[idx].grid(True, alpha=0.3)
    
    # Calculate correlation
    corr = df[['egfrc', mean_col]].corr().iloc[0, 1]
    axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', 
                   transform=axes[idx].transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

### Cell 4: Extraction Ratio Analysis
```python
# Calculate and display extraction ratios
print("EXTRACTION RATIO ANALYSIS")
print("="*80)

def calc_extraction_ratio(p_ra, p_rv):
    """Calculate extraction ratio E = (P_RA - P_RV) / P_RA"""
    if pd.isna(p_ra) or pd.isna(p_rv) or p_ra == 0:
        return None
    return (p_ra - p_rv) / p_ra

# Calculate extraction ratios
df['E_arterial_left'] = df.apply(lambda row: calc_extraction_ratio(
    row['arterial_left_kidney_artery'], row['arterial_left_kidney_vein']), axis=1)
df['E_arterial_right'] = df.apply(lambda row: calc_extraction_ratio(
    row['arterial_right_kidney_artery'], row['arterial_right_kidney_vein']), axis=1)

df['E_venous_left'] = df.apply(lambda row: calc_extraction_ratio(
    row['venous_left_kidney_artery'], row['venous_left_kidney_vein']), axis=1)
df['E_venous_right'] = df.apply(lambda row: calc_extraction_ratio(
    row['venous_right_kidney_artery'], row['venous_right_kidney_vein']), axis=1)

df['E_late_left'] = df.apply(lambda row: calc_extraction_ratio(
    row['late_left_kidney_artery'], row['late_left_kidney_vein']), axis=1)
df['E_late_right'] = df.apply(lambda row: calc_extraction_ratio(
    row['late_right_kidney_artery'], row['late_right_kidney_vein']), axis=1)

# Display statistics
print("\nExtraction Ratio Statistics:")
print("-"*80)

for phase in ['arterial', 'venous', 'late']:
    left_col = f'E_{phase}_left'
    right_col = f'E_{phase}_right'
    
    print(f"\n{phase.upper()} Phase:")
    print(f"  Left:  Mean = {df[left_col].mean():.3f}, Range = [{df[left_col].min():.3f}, {df[left_col].max():.3f}]")
    print(f"  Right: Mean = {df[right_col].mean():.3f}, Range = [{df[right_col].min():.3f}, {df[right_col].max():.3f}]")

# Note about negative values
print("\n" + "="*80)
print("NOTE: Negative extraction ratios occur when P_RV > P_RA")
print("This may indicate:")
print("  - Measurement artifacts")
print("  - Recirculation effects")
print("  - Phase-specific physiology")
print("  - Need for quality control filtering")
```

### Cell 5: Phase Comparison
```python
# Compare vGFR across phases
import seaborn as sns

# Prepare data for visualization
vgfr_data = df[['record_id', 'vgfr_arterial_mean', 'vgfr_venous_mean', 'vgfr_late_mean']].copy()
vgfr_data_melted = vgfr_data.melt(id_vars='record_id', 
                                   var_name='Phase', 
                                   value_name='vGFR')
vgfr_data_melted['Phase'] = vgfr_data_melted['Phase'].str.replace('vgfr_', '').str.replace('_mean', '')

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=vgfr_data_melted, x='Phase', y='vGFR', palette='Set2')
plt.xlabel('Phase', fontsize=12)
plt.ylabel('vGFR (mL/min)', fontsize=12)
plt.title('vGFR Distribution Across Phases', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Statistical comparison
print("\nPhase Comparison:")
print("="*80)
print(vgfr_data_melted.groupby('Phase')['vGFR'].describe())
```

---

## ⚠️ Important Notes

1. **Negative vGFR values**: Can occur when P_RV > P_RA (extraction ratio is negative)
   - This may indicate measurement artifacts or phase-specific physiology
   - Consider filtering or quality control for analysis

2. **RPF value**: Currently fixed at 600 mL/min
   - This is a literature-based estimate for a standard adult
   - Future improvements could adjust based on patient characteristics

3. **Phase differences**: 
   - Arterial phase typically shows highest vGFR (most extraction)
   - Venous and late phases may show different patterns
   - Negative values more common in later phases

4. **Clinical interpretation**: 
   - vGFR provides kidney-specific and phase-specific measurements
   - Can be compared with eGFRc for validation
   - Useful for assessing individual kidney function

---

## 🔄 Re-running the Script

To update the table with new data or modified RPF:

```bash
conda activate nrrd-viewer
python scripts/02_create_gold_anon_segmentations_with_egfr.py
```

The script will automatically calculate all vGFR values.

---

## 📚 Reference

[1] Based on the extraction ratio method for measuring renal function using contrast-enhanced CT imaging.
