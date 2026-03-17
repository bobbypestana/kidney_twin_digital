"""
Create gold.segmentations_with_egfr table by adding eGFRc and vGFR calculations.
Matches record_id from anon_segmentations_with_egfr with case_number.
"""

import duckdb
import pandas as pd
from pathlib import Path

# Database path
db_path = Path(r'c:\Users\FilipeFurlanBellotti\OneDrive - Kvantify\Kvantify - WP5 - Health Sector Use Case\DanQ-WP5 Kvantify-HGH\data_analisys\database\egfr_data.duckdb')

print(f"Database file: {db_path}")
print("=" * 80)

# Connect to database
conn = duckdb.connect(str(db_path))

try:
    # Load the existing gold.segmentations table
    print("\nLoading gold.segmentations...")
    seg_df = conn.execute('SELECT * FROM gold.segmentations').df()
    print(f"  ✓ Loaded {len(seg_df)} records")
    
    # Load anon_segmentations_with_egfr to get eGFRc
    print("\nLoading gold.anon_segmentations_with_egfr...")
    anon_df = conn.execute('SELECT record_id, egfrc, serum_creatinine, current_age FROM gold.anon_segmentations_with_egfr').df()
    print(f"  ✓ Loaded {len(anon_df)} records")
    
    # Merge on record_id = case_number
    print("\n" + "=" * 80)
    print("MERGING SEGMENTATIONS WITH eGFRc")
    print("=" * 80)
    
    # Convert case_number to int for matching
    seg_df['case_number'] = seg_df['case_number'].astype(int)
    anon_df['record_id'] = anon_df['record_id'].astype(int)
    
    # Merge
    result_df = seg_df.merge(
        anon_df[['record_id', 'egfrc', 'serum_creatinine', 'current_age']], 
        left_on='case_number', 
        right_on='record_id', 
        how='left'
    )
    
    # Drop the duplicate record_id column
    if 'record_id' in result_df.columns:
        result_df = result_df.drop(columns=['record_id'])
    
    print(f"\n✓ Merged {len(result_df)} records")
    print(f"  - With eGFRc data: {result_df['egfrc'].notna().sum()}")
    print(f"  - Without eGFRc data: {result_df['egfrc'].isna().sum()}")
    
    if result_df['egfrc'].notna().sum() > 0:
        print(f"\neGFRc statistics:")
        print(f"  - Mean: {result_df['egfrc'].mean():.2f}")
        print(f"  - Median: {result_df['egfrc'].median():.2f}")
        print(f"  - Min: {result_df['egfrc'].min():.2f}")
        print(f"  - Max: {result_df['egfrc'].max():.2f}")
        print(f"  - Std Dev: {result_df['egfrc'].std():.2f}")
    
    # Calculate vGFR (HU method only, no corrections)
    print("\n" + "=" * 80)
    print("CALCULATING vGFR (HU method, no corrections)")
    print("=" * 80)
    
    RPF = 600.0  # mL/min
    print(f"Using fixed RPF: {RPF} mL/min")
    
    def calculate_vgfr(p_ra, p_rv, rpf=RPF):
        """Calculate volumetric GFR using extraction ratio"""
        if pd.isna(p_ra) or pd.isna(p_rv):
            return None
        
        if p_ra == 0:
            return None
        
        E = (p_ra - p_rv) / p_ra
        vgfr = rpf * E
        
        return vgfr
    
    # Arterial phase
    result_df['vgfr_arterial_left'] = result_df.apply(
        lambda row: calculate_vgfr(row['arterial_left_kidney_artery'], 
                                   row['arterial_left_kidney_vein']), axis=1)
    result_df['vgfr_arterial_right'] = result_df.apply(
        lambda row: calculate_vgfr(row['arterial_right_kidney_artery'], 
                                   row['arterial_right_kidney_vein']), axis=1)
    
    # Venous phase
    result_df['vgfr_venous_left'] = result_df.apply(
        lambda row: calculate_vgfr(row['venous_left_kidney_artery'], 
                                   row['venous_left_kidney_vein']), axis=1)
    result_df['vgfr_venous_right'] = result_df.apply(
        lambda row: calculate_vgfr(row['venous_right_kidney_artery'], 
                                   row['venous_right_kidney_vein']), axis=1)
    
    # Late phase
    result_df['vgfr_late_left'] = result_df.apply(
        lambda row: calculate_vgfr(row['late_left_kidney_artery'], 
                                   row['late_left_kidney_vein']), axis=1)
    result_df['vgfr_late_right'] = result_df.apply(
        lambda row: calculate_vgfr(row['late_right_kidney_artery'], 
                                   row['late_right_kidney_vein']), axis=1)
    
    # Mean vGFR per phase
    result_df['vgfr_arterial_mean'] = result_df[['vgfr_arterial_left', 'vgfr_arterial_right']].mean(axis=1)
    result_df['vgfr_venous_mean'] = result_df[['vgfr_venous_left', 'vgfr_venous_right']].mean(axis=1)
    result_df['vgfr_late_mean'] = result_df[['vgfr_late_left', 'vgfr_late_right']].mean(axis=1)
    
    print(f"\n✓ Calculated vGFR for all phases")
    print(f"  - Arterial: {result_df['vgfr_arterial_mean'].notna().sum()} records")
    print(f"  - Venous: {result_df['vgfr_venous_mean'].notna().sum()} records")
    print(f"  - Late: {result_df['vgfr_late_mean'].notna().sum()} records")
    
    if result_df['vgfr_arterial_mean'].notna().sum() > 0:
        print(f"\nvGFR Statistics (mL/min):")
        print(f"\nArterial Phase:")
        print(f"  - Mean: {result_df['vgfr_arterial_mean'].mean():.2f}")
        print(f"  - Range: [{result_df['vgfr_arterial_mean'].min():.2f}, {result_df['vgfr_arterial_mean'].max():.2f}]")
        
        print(f"\nVenous Phase:")
        print(f"  - Mean: {result_df['vgfr_venous_mean'].mean():.2f}")
        print(f"  - Range: [{result_df['vgfr_venous_mean'].min():.2f}, {result_df['vgfr_venous_mean'].max():.2f}]")
        
        print(f"\nLate Phase:")
        print(f"  - Mean: {result_df['vgfr_late_mean'].mean():.2f}")
        print(f"  - Range: [{result_df['vgfr_late_mean'].min():.2f}, {result_df['vgfr_late_mean'].max():.2f}]")
    
    # Create/replace the gold table
    print("\n" + "=" * 80)
    print("CREATING GOLD TABLE")
    print("=" * 80)
    
    # Drop existing table if it exists
    conn.execute('DROP TABLE IF EXISTS gold.segmentations_with_egfr')
    
    # Register DataFrame and create table
    conn.register('result_temp', result_df)
    conn.execute('CREATE TABLE gold.segmentations_with_egfr AS SELECT * FROM result_temp')
    
    print(f"\n✓ Created table: gold.segmentations_with_egfr")
    print(f"✓ Total records: {len(result_df)}")
    print(f"✓ Total columns: {len(result_df.columns)}")
    
    # Display sample
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    sample = result_df[['case_number', 'egfrc', 'vgfr_arterial_mean', 
                        'vgfr_venous_mean', 'vgfr_late_mean']].head(10)
    print(sample.to_string())
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Gold table: gold.segmentations_with_egfr")
    print(f"✓ Total records: {len(result_df)}")
    print(f"✓ Total columns: {len(result_df.columns)}")
    print(f"✓ Records with eGFRc data: {result_df['egfrc'].notna().sum()}")
    print(f"✓ Records without eGFRc data: {result_df['egfrc'].isna().sum()}")
    print(f"\n✓ Data successfully merged to gold layer!")
    
finally:
    conn.close()
