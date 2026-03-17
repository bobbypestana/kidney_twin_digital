# Updated Gold Table with eGFRc

## Summary

The gold table `gold.anon_segmentations_with_egfr` now includes a calculated **eGFRc** column.

### eGFRc Formula

The eGFRc (estimated Glomerular Filtration Rate calculated) is computed using:

```
eGFRc = 141 × (PCC/(0.9×88.4))^α × 0.993^β
```

Where:
- **PCC** = `serum_creatinine` (in μmol/L)
- **α** = -0.411 if PCC ≤ 80 μmol/L, else -1.209
- **β** = `current_age` (in years)

### Table Information

- **Table Name**: `gold.anon_segmentations_with_egfr`
- **Total Records**: 25
- **Total Columns**: 43 (including the new `egfrc` column)

### eGFRc Statistics

Based on the current data:
- **Mean**: 95.61 mL/min/1.73m²
- **Median**: ~95-100 mL/min/1.73m²
- **Min**: 47.65 mL/min/1.73m²
- **Max**: 124.46 mL/min/1.73m²
- **Std Dev**: 15.79 mL/min/1.73m²

---

## Notebook Cell Code

### Cell 1: Load data with eGFRc
```python
import duckdb
import pandas as pd
from pathlib import Path

# Connect to database
db_path = Path('../database/egfr_data.duckdb')
conn = duckdb.connect(str(db_path), read_only=True)

# Load the gold table with eGFRc
anon_seg_egfr_df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').df()

conn.close()

print(f"✓ Loaded {len(anon_seg_egfr_df)} records")
print(f"✓ Total columns: {len(anon_seg_egfr_df.columns)}")
print(f"\nColumns include: {', '.join(anon_seg_egfr_df.columns[:10])}...")

# Display eGFRc statistics
print("\n" + "="*80)
print("eGFRc STATISTICS")
print("="*80)
print(f"Mean eGFRc: {anon_seg_egfr_df['egfrc'].mean():.2f} mL/min/1.73m²")
print(f"Median eGFRc: {anon_seg_egfr_df['egfrc'].median():.2f} mL/min/1.73m²")
print(f"Min eGFRc: {anon_seg_egfr_df['egfrc'].min():.2f} mL/min/1.73m²")
print(f"Max eGFRc: {anon_seg_egfr_df['egfrc'].max():.2f} mL/min/1.73m²")
print(f"Std Dev: {anon_seg_egfr_df['egfrc'].std():.2f} mL/min/1.73m²")
```

### Cell 2: Compare eGFR value vs eGFRc
```python
# Display comparison of reported eGFR vs calculated eGFRc
comparison = anon_seg_egfr_df[['record_id', 'current_age', 'serum_creatinine', 
                                'egfr_value', 'egfrc']].copy()

print("Comparison: Reported eGFR vs Calculated eGFRc")
print("="*80)
display(comparison)

# Note: egfr_value may contain ">90" which is categorical, while egfrc is numeric
```

### Cell 3: Explore specific record with eGFRc
```python
# Set the record_id you want to explore
record_id_to_explore = 22  # Change this value

# Filter data
record_data = anon_seg_egfr_df[anon_seg_egfr_df['record_id'] == record_id_to_explore]

if len(record_data) > 0:
    print(f"Record ID: {record_id_to_explore}")
    print("="*80)
    
    print(f"\nPatient Info:")
    print(f"  Age: {record_data['current_age'].values[0]} years")
    print(f"  Sex: {record_data['sex'].values[0]}")
    
    print(f"\nKidney Function:")
    print(f"  Serum Creatinine: {record_data['serum_creatinine'].values[0]} μmol/L")
    print(f"  Reported eGFR: {record_data['egfr_value'].values[0]}")
    print(f"  Calculated eGFRc: {record_data['egfrc'].values[0]:.2f} mL/min/1.73m²")
    
    print(f"\nKidney Measurements (Arterial):")
    print(f"  Left Kidney Artery: {record_data['arterial_left_kidney_artery'].values[0]}")
    print(f"  Right Kidney Artery: {record_data['arterial_right_kidney_artery'].values[0]}")
else:
    print(f"No data found for record_id {record_id_to_explore}")
```

### Cell 4: Correlation between eGFRc and kidney measurements
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: eGFRc vs Left Kidney Artery (Arterial)
axes[0, 0].scatter(anon_seg_egfr_df['egfrc'], 
                   anon_seg_egfr_df['arterial_left_kidney_artery'], 
                   alpha=0.6, s=100)
axes[0, 0].set_xlabel('eGFRc (mL/min/1.73m²)', fontsize=12)
axes[0, 0].set_ylabel('Left Kidney Artery (Arterial)', fontsize=12)
axes[0, 0].set_title('eGFRc vs Left Kidney Artery', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: eGFRc vs Right Kidney Artery (Arterial)
axes[0, 1].scatter(anon_seg_egfr_df['egfrc'], 
                   anon_seg_egfr_df['arterial_right_kidney_artery'], 
                   alpha=0.6, s=100, color='orange')
axes[0, 1].set_xlabel('eGFRc (mL/min/1.73m²)', fontsize=12)
axes[0, 1].set_ylabel('Right Kidney Artery (Arterial)', fontsize=12)
axes[0, 1].set_title('eGFRc vs Right Kidney Artery', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: eGFRc vs Serum Creatinine
axes[1, 0].scatter(anon_seg_egfr_df['serum_creatinine'], 
                   anon_seg_egfr_df['egfrc'], 
                   alpha=0.6, s=100, color='green')
axes[1, 0].set_xlabel('Serum Creatinine (μmol/L)', fontsize=12)
axes[1, 0].set_ylabel('eGFRc (mL/min/1.73m²)', fontsize=12)
axes[1, 0].set_title('Serum Creatinine vs eGFRc', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: eGFRc vs Age
axes[1, 1].scatter(anon_seg_egfr_df['current_age'], 
                   anon_seg_egfr_df['egfrc'], 
                   alpha=0.6, s=100, color='red')
axes[1, 1].set_xlabel('Age (years)', fontsize=12)
axes[1, 1].set_ylabel('eGFRc (mL/min/1.73m²)', fontsize=12)
axes[1, 1].set_title('Age vs eGFRc', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate correlations
print("\nCorrelation with eGFRc:")
print("="*80)
correlations = {
    'Serum Creatinine': anon_seg_egfr_df[['egfrc', 'serum_creatinine']].corr().iloc[0, 1],
    'Age': anon_seg_egfr_df[['egfrc', 'current_age']].corr().iloc[0, 1],
    'Left Kidney Artery': anon_seg_egfr_df[['egfrc', 'arterial_left_kidney_artery']].corr().iloc[0, 1],
    'Right Kidney Artery': anon_seg_egfr_df[['egfrc', 'arterial_right_kidney_artery']].corr().iloc[0, 1],
}

for name, corr in correlations.items():
    print(f"{name:25s}: {corr:7.3f}")
```

### Cell 5: eGFRc Distribution
```python
import matplotlib.pyplot as plt

# Create histogram of eGFRc values
plt.figure(figsize=(10, 6))
plt.hist(anon_seg_egfr_df['egfrc'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('eGFRc (mL/min/1.73m²)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Calculated eGFRc Values', fontsize=14)
plt.axvline(anon_seg_egfr_df['egfrc'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {anon_seg_egfr_df["egfrc"].mean():.2f}')
plt.axvline(anon_seg_egfr_df['egfrc'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Median: {anon_seg_egfr_df["egfrc"].median():.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Show eGFR categories
print("\neGFRc Categories (CKD Stages):")
print("="*80)

def categorize_egfr(egfrc):
    if egfrc >= 90:
        return 'Stage 1: Normal (≥90)'
    elif egfrc >= 60:
        return 'Stage 2: Mild (60-89)'
    elif egfrc >= 45:
        return 'Stage 3a: Mild-Moderate (45-59)'
    elif egfrc >= 30:
        return 'Stage 3b: Moderate-Severe (30-44)'
    elif egfrc >= 15:
        return 'Stage 4: Severe (15-29)'
    else:
        return 'Stage 5: Kidney Failure (<15)'

anon_seg_egfr_df['ckd_stage'] = anon_seg_egfr_df['egfrc'].apply(categorize_egfr)
print(anon_seg_egfr_df['ckd_stage'].value_counts().sort_index())
```

---

## Key Points

1. **eGFRc is now automatically calculated** when you run the script
2. The formula accounts for:
   - Different α values based on creatinine threshold (80 μmol/L)
   - Age-based adjustment using exponential decay (0.993^age)
3. **Units**: eGFRc is in mL/min/1.73m²
4. The calculation requires both `serum_creatinine` and `current_age` to be non-null

## Re-running the Script

To recreate the table with updated data:
```bash
python scripts/02_create_gold_anon_segmentations_with_egfr.py
```

The script will automatically:
1. Merge segmentations with eGFR data
2. Find closest eGFR date to scan date
3. Calculate eGFRc for all records
4. Create the gold table with all columns
