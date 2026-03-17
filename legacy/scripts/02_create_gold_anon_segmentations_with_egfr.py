"""
Create Gold Layer Anonymized Segmentations with eGFR

This script performs a left join between bronze.anon_segmentations and bronze.anon_egfr,
matching on record_id and finding the closest egfr_date to scan_date.

The result is stored in gold.anon_segmentations_with_egfr.
"""

import duckdb
import pandas as pd
from pathlib import Path


def main():
    """Main function to create gold layer anonymized segmentations with eGFR."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent
    db_file = data_dir / "database" / "egfr_data.duckdb"
    
    print(f"Database file: {db_file}\n")
    
    # Connect to database
    conn = duckdb.connect(str(db_file))
    
    try:
        # Create gold schema if it doesn't exist
        print("=" * 80)
        print("CREATING GOLD SCHEMA")
        print("=" * 80)
        conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
        print("Created schema 'gold'\n")
        
        # Check source tables
        print("=" * 80)
        print("CHECKING SOURCE TABLES")
        print("=" * 80)
        
        seg_count = conn.execute("SELECT COUNT(*) FROM bronze.anon_segmentations").fetchone()[0]
        egfr_count = conn.execute("SELECT COUNT(*) FROM bronze.anon_egfr").fetchone()[0]
        
        print(f"bronze.anon_segmentations: {seg_count} rows")
        print(f"bronze.anon_egfr: {egfr_count} rows\n")
        
        # Perform the join with closest date matching
        print("=" * 80)
        print("PERFORMING LEFT JOIN WITH CLOSEST DATE MATCHING")
        print("=" * 80)
        
        # SQL query to find the closest egfr_date for each scan_date
        query = """
        WITH ranked_egfr AS (
            SELECT 
                seg.record_id,
                seg.current_age,
                seg.sex,
                seg.scan_date,
                seg.arterial_left_kidney_artery,
                seg.arterial_left_kidney_vein,
                seg.arterial_right_kidney_artery,
                seg.arterial_right_kidney_vein,
                seg.arterial_aorta,
                seg.arterial_venacava_below_kidney,
                seg.arterial_venecava_between_kidney_hepatic,
                seg.arterial_venecava_above_hepatic,
                seg.arterial_right_hepatic_vein,
                seg.arterial_left_hepatic_vein,
                seg.arterial_portal_vein,
                seg.venous_left_kidney_artery,
                seg.venous_left_kidney_vein,
                seg.venous_right_kidney_artery,
                seg.venous_right_kidney_vein,
                seg.venous_aorta,
                seg.venous_venacava_below_kidney,
                seg.venous_venacava_between_kidney_hepatic,
                seg.venous_venacava_above_hepatic,
                seg.venous_right_hepatic_vein,
                seg.venous_left_hepatic_vein,
                seg.venous_portal_vein,
                seg.late_left_kidney_artery,
                seg.late_left_kidney_vein,
                seg.late_right_kidney_vein,
                seg.late_right_kidney_artery,
                seg.late_aorta,
                seg.late_venacava_below_kidney,
                seg.late_venacava_between_kidney_hepatic,
                seg.late_venacava_above_hepatic,
                seg.late_right_hepatic_vein,
                seg.late_left_hepatic_vein,
                seg.late_portal_vein,
                egfr.egfr_date,
                egfr.egfr_value,
                egfr.serum_creatinine,
                egfr.redcap_repeat_instance,
                -- Calculate absolute date difference in days
                ABS(DATEDIFF('day', 
                    STRPTIME(seg.scan_date, '%d-%m-%Y'),
                    STRPTIME(egfr.egfr_date, '%d-%m-%Y')
                )) AS date_diff_days,
                -- Rank by date difference (closest first)
                ROW_NUMBER() OVER (
                    PARTITION BY seg.record_id, seg.scan_date 
                    ORDER BY ABS(DATEDIFF('day', 
                        STRPTIME(seg.scan_date, '%d-%m-%Y'),
                        STRPTIME(egfr.egfr_date, '%d-%m-%Y')
                    ))
                ) AS rn
            FROM bronze.anon_segmentations seg
            LEFT JOIN bronze.anon_egfr egfr
                ON seg.record_id = egfr.record_id
        )
        SELECT 
            record_id,
            current_age,
            sex,
            scan_date,
            arterial_left_kidney_artery,
            arterial_left_kidney_vein,
            arterial_right_kidney_artery,
            arterial_right_kidney_vein,
            arterial_aorta,
            arterial_venacava_below_kidney,
            arterial_venecava_between_kidney_hepatic,
            arterial_venecava_above_hepatic,
            arterial_right_hepatic_vein,
            arterial_left_hepatic_vein,
            arterial_portal_vein,
            venous_left_kidney_artery,
            venous_left_kidney_vein,
            venous_right_kidney_artery,
            venous_right_kidney_vein,
            venous_aorta,
            venous_venacava_below_kidney,
            venous_venacava_between_kidney_hepatic,
            venous_venacava_above_hepatic,
            venous_right_hepatic_vein,
            venous_left_hepatic_vein,
            venous_portal_vein,
            late_left_kidney_artery,
            late_left_kidney_vein,
            late_right_kidney_vein,
            late_right_kidney_artery,
            late_aorta,
            late_venacava_below_kidney,
            late_venacava_between_kidney_hepatic,
            late_venacava_above_hepatic,
            late_right_hepatic_vein,
            late_left_hepatic_vein,
            late_portal_vein,
            egfr_date,
            egfr_value,
            serum_creatinine,
            redcap_repeat_instance,
            date_diff_days
        FROM ranked_egfr
        WHERE rn = 1 OR rn IS NULL
        ORDER BY record_id
        """
        
        # Execute query and get result
        result_df = conn.execute(query).df()
        
        print(f"Joined data shape: {result_df.shape}")
        print(f"Total rows: {len(result_df)}")
        
        # Show matching statistics
        matched = result_df['egfr_date'].notna().sum()
        unmatched = result_df['egfr_date'].isna().sum()
        
        print(f"\nMatching statistics:")
        print(f"  - Matched with eGFR: {matched} ({matched/len(result_df)*100:.1f}%)")
        print(f"  - No eGFR match: {unmatched} ({unmatched/len(result_df)*100:.1f}%)")
        
        if matched > 0:
            print(f"\nDate difference statistics (for matched records):")
            print(f"  - Mean: {result_df[result_df['date_diff_days'].notna()]['date_diff_days'].mean():.1f} days")
            print(f"  - Median: {result_df[result_df['date_diff_days'].notna()]['date_diff_days'].median():.1f} days")
            print(f"  - Min: {result_df[result_df['date_diff_days'].notna()]['date_diff_days'].min():.0f} days")
            print(f"  - Max: {result_df[result_df['date_diff_days'].notna()]['date_diff_days'].max():.0f} days")
        
        # Calculate eGFRc
        print("\n" + "=" * 80)
        print("CALCULATING eGFRc")
        print("=" * 80)
        
        def calculate_egfrc(row):
            """
            Calculate eGFRc using the formula:
            eGFRc = 141 * (PCC/(0.9*88.4))^α * 0.993^β
            
            Where:
            - PCC = serum_creatinine (μmol/L)
            - α = -0.411 if PCC ≤ 80 μmol/L, else -1.209
            - β = age (years)
            """
            pcc = row['serum_creatinine']
            age = row['current_age']
            
            # If either value is missing, return None
            if pd.isna(pcc) or pd.isna(age):
                return None
            
            # Determine alpha based on PCC threshold
            alpha = -0.411 if pcc <= 80 else -1.209
            
            # Calculate eGFRc
            # eGFRc = 141 * (PCC/(0.9*88.4))^α * 0.993^β
            egfrc = 141 * ((pcc / (0.9 * 88.4)) ** alpha) * (0.993 ** age)
            
            return egfrc
        
        # Apply calculation to all rows
        result_df['egfrc'] = result_df.apply(calculate_egfrc, axis=1)
        
        # Show eGFRc statistics
        egfrc_calculated = result_df['egfrc'].notna().sum()
        print(f"eGFRc calculated for {egfrc_calculated} records")
        
        if egfrc_calculated > 0:
            print(f"\neGFRc statistics:")
            print(f"  - Mean: {result_df['egfrc'].mean():.2f}")
            print(f"  - Median: {result_df['egfrc'].median():.2f}")
            print(f"  - Min: {result_df['egfrc'].min():.2f}")
            print(f"  - Max: {result_df['egfrc'].max():.2f}")
            print(f"  - Std Dev: {result_df['egfrc'].std():.2f}")
        
        # HU to Concentration Conversion
        print("\n" + "=" * 80)
        print("CONVERTING HU TO IODINE CONCENTRATION")
        print("=" * 80)
        
        # Conversion parameters
        K = 25  # HU per mg/mL (assuming 120 kVp)
        HU_BASELINE_BLOOD = 45  # Literature-based baseline for blood vessels
        
        print(f"Conversion factor (k): {K} HU per mg/mL")
        print(f"Baseline HU for blood: {HU_BASELINE_BLOOD}")
        
        def hu_to_concentration(hu, baseline=0, k=K):
            """
            Convert Hounsfield Units to iodine concentration.
            
            Parameters:
            - hu: Measured HU value
            - baseline: Baseline HU to subtract
            - k: Conversion factor (HU per mg/mL)
            
            Returns:
            - Iodine concentration in mg/mL
            """
            if pd.isna(hu):
                return None
            return (hu - baseline) / k
        
        # OPTION 2: Literature-based baseline (HU = 45 for blood)
        print("\nOption 2: Converting with literature-based baseline (HU = 45)...")
        
        # Arterial phase
        result_df['conc_lit_arterial_left_artery'] = result_df['arterial_left_kidney_artery'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_arterial_left_vein'] = result_df['arterial_left_kidney_vein'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_arterial_right_artery'] = result_df['arterial_right_kidney_artery'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_arterial_right_vein'] = result_df['arterial_right_kidney_vein'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        
        # Venous phase
        result_df['conc_lit_venous_left_artery'] = result_df['venous_left_kidney_artery'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_venous_left_vein'] = result_df['venous_left_kidney_vein'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_venous_right_artery'] = result_df['venous_right_kidney_artery'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_venous_right_vein'] = result_df['venous_right_kidney_vein'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        
        # Late phase
        result_df['conc_lit_late_left_artery'] = result_df['late_left_kidney_artery'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_late_left_vein'] = result_df['late_left_kidney_vein'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_late_right_artery'] = result_df['late_right_kidney_artery'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        result_df['conc_lit_late_right_vein'] = result_df['late_right_kidney_vein'].apply(
            lambda hu: hu_to_concentration(hu, baseline=HU_BASELINE_BLOOD))
        
        print(f"  ✓ Created 12 concentration columns with literature baseline")
        
        # OPTION 3: Late phase as pseudo-baseline (patient-specific)
        print("\nOption 3: Converting with late phase as pseudo-baseline...")
        
        # Arterial phase (using late phase as baseline)
        result_df['conc_late_arterial_left_artery'] = result_df.apply(
            lambda row: hu_to_concentration(row['arterial_left_kidney_artery'], 
                                           baseline=row['late_left_kidney_artery'] if pd.notna(row['late_left_kidney_artery']) else 0), axis=1)
        result_df['conc_late_arterial_left_vein'] = result_df.apply(
            lambda row: hu_to_concentration(row['arterial_left_kidney_vein'], 
                                           baseline=row['late_left_kidney_vein'] if pd.notna(row['late_left_kidney_vein']) else 0), axis=1)
        result_df['conc_late_arterial_right_artery'] = result_df.apply(
            lambda row: hu_to_concentration(row['arterial_right_kidney_artery'], 
                                           baseline=row['late_right_kidney_artery'] if pd.notna(row['late_right_kidney_artery']) else 0), axis=1)
        result_df['conc_late_arterial_right_vein'] = result_df.apply(
            lambda row: hu_to_concentration(row['arterial_right_kidney_vein'], 
                                           baseline=row['late_right_kidney_vein'] if pd.notna(row['late_right_kidney_vein']) else 0), axis=1)
        
        # Venous phase (using late phase as baseline)
        result_df['conc_late_venous_left_artery'] = result_df.apply(
            lambda row: hu_to_concentration(row['venous_left_kidney_artery'], 
                                           baseline=row['late_left_kidney_artery'] if pd.notna(row['late_left_kidney_artery']) else 0), axis=1)
        result_df['conc_late_venous_left_vein'] = result_df.apply(
            lambda row: hu_to_concentration(row['venous_left_kidney_vein'], 
                                           baseline=row['late_left_kidney_vein'] if pd.notna(row['late_left_kidney_vein']) else 0), axis=1)
        result_df['conc_late_venous_right_artery'] = result_df.apply(
            lambda row: hu_to_concentration(row['venous_right_kidney_artery'], 
                                           baseline=row['late_right_kidney_artery'] if pd.notna(row['late_right_kidney_artery']) else 0), axis=1)
        result_df['conc_late_venous_right_vein'] = result_df.apply(
            lambda row: hu_to_concentration(row['venous_right_kidney_vein'], 
                                           baseline=row['late_right_kidney_vein'] if pd.notna(row['late_right_kidney_vein']) else 0), axis=1)
        
        print(f"  ✓ Created 8 concentration columns with late phase baseline")
        
        # Calculate vGFR
        print("\n" + "=" * 80)
        print("CALCULATING vGFR (Volumetric GFR)")
        print("=" * 80)
        
        # Fixed RPF value (mL/min) - literature-based estimate for standard adult
        RPF = 600.0
        print(f"Using fixed RPF: {RPF} mL/min")
        
        def calculate_vgfr(p_ra, p_rv, rpf=RPF):
            """
            Calculate volumetric GFR using extraction ratio.
            
            vGFR = RPF × E
            where E = (P_RA - P_RV) / P_RA
            
            Therefore: vGFR = RPF × (P_RA - P_RV) / P_RA
            
            Parameters:
            - p_ra: Iodine concentration in renal artery
            - p_rv: Iodine concentration in renal vein
            - rpf: Renal plasma flow (mL/min)
            
            Returns:
            - vGFR in mL/min, or None if data is missing or invalid
            """
            if pd.isna(p_ra) or pd.isna(p_rv):
                return None
            
            # Avoid division by zero
            if p_ra == 0:
                return None
            
            # Calculate extraction ratio
            E = (p_ra - p_rv) / p_ra
            
            # Calculate vGFR
            vgfr = rpf * E
            
            return vgfr
        
        # Calculate vGFR for each kidney and phase
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
        
        # Calculate mean vGFR for each phase (average of left and right)
        result_df['vgfr_arterial_mean'] = result_df[['vgfr_arterial_left', 'vgfr_arterial_right']].mean(axis=1)
        result_df['vgfr_venous_mean'] = result_df[['vgfr_venous_left', 'vgfr_venous_right']].mean(axis=1)
        result_df['vgfr_late_mean'] = result_df[['vgfr_late_left', 'vgfr_late_right']].mean(axis=1)
        
        print(f"\n✓ Calculated vGFR using raw HU values (original method)")
        
        # Calculate vGFR using LITERATURE-BASED concentrations
        print(f"\nCalculating vGFR using literature-based concentrations...")
        
        # Arterial phase
        result_df['vgfr_lit_arterial_left'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_lit_arterial_left_artery'], 
                                       row['conc_lit_arterial_left_vein']), axis=1)
        result_df['vgfr_lit_arterial_right'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_lit_arterial_right_artery'], 
                                       row['conc_lit_arterial_right_vein']), axis=1)
        
        # Venous phase
        result_df['vgfr_lit_venous_left'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_lit_venous_left_artery'], 
                                       row['conc_lit_venous_left_vein']), axis=1)
        result_df['vgfr_lit_venous_right'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_lit_venous_right_artery'], 
                                       row['conc_lit_venous_right_vein']), axis=1)
        
        # Late phase
        result_df['vgfr_lit_late_left'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_lit_late_left_artery'], 
                                       row['conc_lit_late_left_vein']), axis=1)
        result_df['vgfr_lit_late_right'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_lit_late_right_artery'], 
                                       row['conc_lit_late_right_vein']), axis=1)
        
        # Mean vGFR per phase (literature-based)
        result_df['vgfr_lit_arterial_mean'] = result_df[['vgfr_lit_arterial_left', 'vgfr_lit_arterial_right']].mean(axis=1)
        result_df['vgfr_lit_venous_mean'] = result_df[['vgfr_lit_venous_left', 'vgfr_lit_venous_right']].mean(axis=1)
        result_df['vgfr_lit_late_mean'] = result_df[['vgfr_lit_late_left', 'vgfr_lit_late_right']].mean(axis=1)
        
        print(f"  ✓ Created 9 vGFR columns using literature baseline")
        
        # Calculate vGFR using LATE PHASE as baseline concentrations
        print(f"\nCalculating vGFR using late phase baseline concentrations...")
        
        # Arterial phase
        result_df['vgfr_late_arterial_left'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_late_arterial_left_artery'], 
                                       row['conc_late_arterial_left_vein']), axis=1)
        result_df['vgfr_late_arterial_right'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_late_arterial_right_artery'], 
                                       row['conc_late_arterial_right_vein']), axis=1)
        
        # Venous phase
        result_df['vgfr_late_venous_left'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_late_venous_left_artery'], 
                                       row['conc_late_venous_left_vein']), axis=1)
        result_df['vgfr_late_venous_right'] = result_df.apply(
            lambda row: calculate_vgfr(row['conc_late_venous_right_artery'], 
                                       row['conc_late_venous_right_vein']), axis=1)
        
        # Mean vGFR per phase (late phase baseline)
        result_df['vgfr_late_arterial_mean'] = result_df[['vgfr_late_arterial_left', 'vgfr_late_arterial_right']].mean(axis=1)
        result_df['vgfr_late_venous_mean'] = result_df[['vgfr_late_venous_left', 'vgfr_late_venous_right']].mean(axis=1)
        
        print(f"  ✓ Created 6 vGFR columns using late phase baseline")

        # Calculate Wavefront Correction Factor (W_pv)
        print("\n" + "=" * 80)
        print("CALCULATING WAVEFRONT CORRECTION FACTOR (W_pv)")
        print("=" * 80)
        print("W_pv = P_portal_vein / P_aorta")
        print("Applied to: vGFR = RPF × (P_RA - P_RV × W_pv) / P_RA")
        
        def calculate_w_pv(p_pv, p_aorta):
            """
            Calculate wavefront correction factor.
            
            W_pv = P_pv / P_aorta
            
            Parameters:
            - p_pv: Portal vein HU/concentration
            - p_aorta: Aorta HU/concentration
            
            Returns:
            - Wavefront correction factor, or None if data missing
            """
            if pd.isna(p_pv) or pd.isna(p_aorta) or p_aorta == 0:
                return None
            return p_pv / p_aorta
        
        # Calculate W_pv for each phase
        result_df['w_pv_arterial'] = result_df.apply(
            lambda row: calculate_w_pv(row['arterial_portal_vein'], row['arterial_aorta']), axis=1)
        result_df['w_pv_venous'] = result_df.apply(
            lambda row: calculate_w_pv(row['venous_portal_vein'], row['venous_aorta']), axis=1)
        result_df['w_pv_late'] = result_df.apply(
            lambda row: calculate_w_pv(row['late_portal_vein'], row['late_aorta']), axis=1)
        
        print(f"\n✓ Calculated W_pv for all phases")
        print(f"  - Arterial: {result_df['w_pv_arterial'].notna().sum()} records")
        print(f"  - Venous: {result_df['w_pv_venous'].notna().sum()} records")
        print(f"  - Late: {result_df['w_pv_late'].notna().sum()} records")
        
        # Show W_pv statistics
        if result_df['w_pv_arterial'].notna().sum() > 0:
            print(f"\nW_pv Statistics:")
            print(f"  Arterial - Mean: {result_df['w_pv_arterial'].mean():.3f}, Range: [{result_df['w_pv_arterial'].min():.3f}, {result_df['w_pv_arterial'].max():.3f}]")
            print(f"  Venous   - Mean: {result_df['w_pv_venous'].mean():.3f}, Range: [{result_df['w_pv_venous'].min():.3f}, {result_df['w_pv_venous'].max():.3f}]")
            print(f"  Late     - Mean: {result_df['w_pv_late'].mean():.3f}, Range: [{result_df['w_pv_late'].min():.3f}, {result_df['w_pv_late'].max():.3f}]")
        
        # Calculate vGFR with wavefront correction (HU method only)
        print(f"\nCalculating vGFR with wavefront correction (HU method)...")
        
        def calculate_vgfr_with_correction(p_ra, p_rv, w_pv, rpf=RPF):
            """
            Calculate vGFR with wavefront correction.
            
            vGFR = RPF × (P_RA - P_RV × W_pv) / P_RA
            
            Parameters:
            - p_ra: Renal artery HU
            - p_rv: Renal vein HU
            - w_pv: Wavefront correction factor
            - rpf: Renal plasma flow
            
            Returns:
            - vGFR with correction, or None if data missing
            """
            if pd.isna(p_ra) or pd.isna(p_rv) or pd.isna(w_pv):
                return None
            
            if p_ra == 0:
                return None
            
            # Apply wavefront correction to renal vein
            p_rv_corrected = p_rv * w_pv
            
            # Calculate extraction ratio with correction
            E_corrected = (p_ra - p_rv_corrected) / p_ra
            
            # Calculate vGFR
            vgfr = rpf * E_corrected
            
            return vgfr
        
        # Arterial phase with correction
        result_df['vgfr_wf_arterial_left'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['arterial_left_kidney_artery'], 
                row['arterial_left_kidney_vein'],
                row['w_pv_arterial']), axis=1)
        result_df['vgfr_wf_arterial_right'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['arterial_right_kidney_artery'], 
                row['arterial_right_kidney_vein'],
                row['w_pv_arterial']), axis=1)
        
        # Venous phase with correction
        result_df['vgfr_wf_venous_left'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['venous_left_kidney_artery'], 
                row['venous_left_kidney_vein'],
                row['w_pv_venous']), axis=1)
        result_df['vgfr_wf_venous_right'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['venous_right_kidney_artery'], 
                row['venous_right_kidney_vein'],
                row['w_pv_venous']), axis=1)
        
        # Late phase with correction
        result_df['vgfr_wf_late_left'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['late_left_kidney_artery'], 
                row['late_left_kidney_vein'],
                row['w_pv_late']), axis=1)
        result_df['vgfr_wf_late_right'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['late_right_kidney_artery'], 
                row['late_right_kidney_vein'],
                row['w_pv_late']), axis=1)
        
        # Mean vGFR with correction
        result_df['vgfr_wf_arterial_mean'] = result_df[['vgfr_wf_arterial_left', 'vgfr_wf_arterial_right']].mean(axis=1)
        result_df['vgfr_wf_venous_mean'] = result_df[['vgfr_wf_venous_left', 'vgfr_wf_venous_right']].mean(axis=1)
        result_df['vgfr_wf_late_mean'] = result_df[['vgfr_wf_late_left', 'vgfr_wf_late_right']].mean(axis=1)
        
        print(f"  ✓ Created 9 vGFR columns with wavefront correction")
        
        # Show wavefront-corrected vGFR statistics
        if result_df['vgfr_wf_arterial_mean'].notna().sum() > 0:
            print(f"\nvGFR with Wavefront Correction Statistics (mL/min):")
            print(f"  Arterial - Mean: {result_df['vgfr_wf_arterial_mean'].mean():.2f}, Range: [{result_df['vgfr_wf_arterial_mean'].min():.2f}, {result_df['vgfr_wf_arterial_mean'].max():.2f}]")
            print(f"  Venous   - Mean: {result_df['vgfr_wf_venous_mean'].mean():.2f}, Range: [{result_df['vgfr_wf_venous_mean'].min():.2f}, {result_df['vgfr_wf_venous_mean'].max():.2f}]")
            print(f"  Late     - Mean: {result_df['vgfr_wf_late_mean'].mean():.2f}, Range: [{result_df['vgfr_wf_late_mean'].min():.2f}, {result_df['vgfr_wf_late_mean'].max():.2f}]")
        
        
        # Calculate Back-Propagated Wavefront Correction Factor (W_back)
        print("\n" + "=" * 80)
        print("CALCULATING BACK-PROPAGATED WAVEFRONT CORRECTION (W_back)")
        print("=" * 80)
        print("W_back = (P_RA * (1 - eGFR/RPF)) / P_RV")
        print("Applied to: vGFR_back = RPF × (P_RA - P_RV × W_back) / P_RA")
        
        def calculate_w_back(p_ra, p_rv, egfr, rpf=RPF):
            """
            Calculate back-propagated wavefront correction factor.
            
            W_back = (P_RA * (1 - eGFR/RPF)) / P_RV
            
            Parameters:
            - p_ra: Renal artery HU/concentration
            - p_rv: Renal vein HU/concentration
            - egfr: Estimated GFR (mL/min)
            - rpf: Renal plasma flow (mL/min)
            
            Returns:
            - Back-propagated wavefront correction factor, or None if data missing
            """
            if pd.isna(p_ra) or pd.isna(p_rv) or pd.isna(egfr):
                return None
            if p_rv == 0 or rpf == 0:
                return None
            
            # Calculate W_back
            w_back = (p_ra * (1 - egfr / rpf)) / p_rv
            return w_back
        
        # Calculate W_back for each phase using egfrc
        result_df['w_back_arterial_left'] = result_df.apply(
            lambda row: calculate_w_back(
                row['arterial_left_kidney_artery'], 
                row['arterial_left_kidney_vein'],
                row['egfrc']), axis=1)
        result_df['w_back_arterial_right'] = result_df.apply(
            lambda row: calculate_w_back(
                row['arterial_right_kidney_artery'], 
                row['arterial_right_kidney_vein'],
                row['egfrc']), axis=1)
        
        result_df['w_back_venous_left'] = result_df.apply(
            lambda row: calculate_w_back(
                row['venous_left_kidney_artery'], 
                row['venous_left_kidney_vein'],
                row['egfrc']), axis=1)
        result_df['w_back_venous_right'] = result_df.apply(
            lambda row: calculate_w_back(
                row['venous_right_kidney_artery'], 
                row['venous_right_kidney_vein'],
                row['egfrc']), axis=1)
        
        result_df['w_back_late_left'] = result_df.apply(
            lambda row: calculate_w_back(
                row['late_left_kidney_artery'], 
                row['late_left_kidney_vein'],
                row['egfrc']), axis=1)
        result_df['w_back_late_right'] = result_df.apply(
            lambda row: calculate_w_back(
                row['late_right_kidney_artery'], 
                row['late_right_kidney_vein'],
                row['egfrc']), axis=1)
        
        print(f"\n✓ Calculated W_back for all phases and kidneys")
        print(f"  - Arterial Left: {result_df['w_back_arterial_left'].notna().sum()} records")
        print(f"  - Arterial Right: {result_df['w_back_arterial_right'].notna().sum()} records")
        print(f"  - Venous Left: {result_df['w_back_venous_left'].notna().sum()} records")
        print(f"  - Venous Right: {result_df['w_back_venous_right'].notna().sum()} records")
        print(f"  - Late Left: {result_df['w_back_late_left'].notna().sum()} records")
        print(f"  - Late Right: {result_df['w_back_late_right'].notna().sum()} records")
        
        # Show W_back statistics
        if result_df['w_back_arterial_left'].notna().sum() > 0:
            print(f"\nW_back Statistics:")
            print(f"  Arterial Left  - Mean: {result_df['w_back_arterial_left'].mean():.3f}, Range: [{result_df['w_back_arterial_left'].min():.3f}, {result_df['w_back_arterial_left'].max():.3f}]")
            print(f"  Arterial Right - Mean: {result_df['w_back_arterial_right'].mean():.3f}, Range: [{result_df['w_back_arterial_right'].min():.3f}, {result_df['w_back_arterial_right'].max():.3f}]")
            print(f"  Venous Left    - Mean: {result_df['w_back_venous_left'].mean():.3f}, Range: [{result_df['w_back_venous_left'].min():.3f}, {result_df['w_back_venous_left'].max():.3f}]")
            print(f"  Venous Right   - Mean: {result_df['w_back_venous_right'].mean():.3f}, Range: [{result_df['w_back_venous_right'].min():.3f}, {result_df['w_back_venous_right'].max():.3f}]")
            print(f"  Late Left      - Mean: {result_df['w_back_late_left'].mean():.3f}, Range: [{result_df['w_back_late_left'].min():.3f}, {result_df['w_back_late_left'].max():.3f}]")
            print(f"  Late Right     - Mean: {result_df['w_back_late_right'].mean():.3f}, Range: [{result_df['w_back_late_right'].min():.3f}, {result_df['w_back_late_right'].max():.3f}]")
        
        # Calculate vGFR with back-propagated wavefront correction
        print(f"\nCalculating vGFR with back-propagated wavefront correction...")
        
        # Arterial phase with W_back correction
        result_df['vgfr_back_arterial_left'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['arterial_left_kidney_artery'], 
                row['arterial_left_kidney_vein'],
                row['w_back_arterial_left']), axis=1)
        result_df['vgfr_back_arterial_right'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['arterial_right_kidney_artery'], 
                row['arterial_right_kidney_vein'],
                row['w_back_arterial_right']), axis=1)
        
        # Venous phase with W_back correction
        result_df['vgfr_back_venous_left'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['venous_left_kidney_artery'], 
                row['venous_left_kidney_vein'],
                row['w_back_venous_left']), axis=1)
        result_df['vgfr_back_venous_right'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['venous_right_kidney_artery'], 
                row['venous_right_kidney_vein'],
                row['w_back_venous_right']), axis=1)
        
        # Late phase with W_back correction
        result_df['vgfr_back_late_left'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['late_left_kidney_artery'], 
                row['late_left_kidney_vein'],
                row['w_back_late_left']), axis=1)
        result_df['vgfr_back_late_right'] = result_df.apply(
            lambda row: calculate_vgfr_with_correction(
                row['late_right_kidney_artery'], 
                row['late_right_kidney_vein'],
                row['w_back_late_right']), axis=1)
        
        # Mean vGFR with back-propagated correction
        result_df['vgfr_back_arterial_mean'] = result_df[['vgfr_back_arterial_left', 'vgfr_back_arterial_right']].mean(axis=1)
        result_df['vgfr_back_venous_mean'] = result_df[['vgfr_back_venous_left', 'vgfr_back_venous_right']].mean(axis=1)
        result_df['vgfr_back_late_mean'] = result_df[['vgfr_back_late_left', 'vgfr_back_late_right']].mean(axis=1)
        
        print(f"  ✓ Created 9 vGFR columns with back-propagated wavefront correction")
        
        # Show back-propagated vGFR statistics
        if result_df['vgfr_back_arterial_mean'].notna().sum() > 0:
            print(f"\nvGFR with Back-Propagated Wavefront Correction Statistics (mL/min):")
            print(f"  Arterial - Mean: {result_df['vgfr_back_arterial_mean'].mean():.2f}, Range: [{result_df['vgfr_back_arterial_mean'].min():.2f}, {result_df['vgfr_back_arterial_mean'].max():.2f}]")
            print(f"  Venous   - Mean: {result_df['vgfr_back_venous_mean'].mean():.2f}, Range: [{result_df['vgfr_back_venous_mean'].min():.2f}, {result_df['vgfr_back_venous_mean'].max():.2f}]")
            print(f"  Late     - Mean: {result_df['vgfr_back_late_mean'].mean():.2f}, Range: [{result_df['vgfr_back_late_mean'].min():.2f}, {result_df['vgfr_back_late_mean'].max():.2f}]")
        
        
        # Show vGFR statistics
        print(f"\nvGFR calculated for:")
        print(f"  - Arterial phase (left): {result_df['vgfr_arterial_left'].notna().sum()} records")
        print(f"  - Arterial phase (right): {result_df['vgfr_arterial_right'].notna().sum()} records")
        print(f"  - Venous phase (left): {result_df['vgfr_venous_left'].notna().sum()} records")
        print(f"  - Venous phase (right): {result_df['vgfr_venous_right'].notna().sum()} records")
        print(f"  - Late phase (left): {result_df['vgfr_late_left'].notna().sum()} records")
        print(f"  - Late phase (right): {result_df['vgfr_late_right'].notna().sum()} records")
        
        # Statistics for mean vGFR by phase
        print(f"\nvGFR Mean Statistics (mL/min):")
        print(f"\nArterial Phase:")
        if result_df['vgfr_arterial_mean'].notna().sum() > 0:
            print(f"  - Mean: {result_df['vgfr_arterial_mean'].mean():.2f}")
            print(f"  - Median: {result_df['vgfr_arterial_mean'].median():.2f}")
            print(f"  - Min: {result_df['vgfr_arterial_mean'].min():.2f}")
            print(f"  - Max: {result_df['vgfr_arterial_mean'].max():.2f}")
        
        print(f"\nVenous Phase:")
        if result_df['vgfr_venous_mean'].notna().sum() > 0:
            print(f"  - Mean: {result_df['vgfr_venous_mean'].mean():.2f}")
            print(f"  - Median: {result_df['vgfr_venous_mean'].median():.2f}")
            print(f"  - Min: {result_df['vgfr_venous_mean'].min():.2f}")
            print(f"  - Max: {result_df['vgfr_venous_mean'].max():.2f}")
        
        print(f"\nLate Phase:")
        if result_df['vgfr_late_mean'].notna().sum() > 0:
            print(f"  - Mean: {result_df['vgfr_late_mean'].mean():.2f}")
            print(f"  - Median: {result_df['vgfr_late_mean'].median():.2f}")
            print(f"  - Min: {result_df['vgfr_late_mean'].min():.2f}")
            print(f"  - Max: {result_df['vgfr_late_mean'].max():.2f}")
        
        # Create gold table
        print("\n" + "=" * 80)
        print("CREATING GOLD TABLE")
        print("=" * 80)
        
        conn.execute("DROP TABLE IF EXISTS gold.anon_segmentations_with_egfr")
        conn.execute("CREATE TABLE gold.anon_segmentations_with_egfr AS SELECT * FROM result_df")
        
        count = conn.execute("SELECT COUNT(*) FROM gold.anon_segmentations_with_egfr").fetchone()[0]
        print(f"Created table 'gold.anon_segmentations_with_egfr' with {count} rows")
        
        # Show sample
        print("\n" + "=" * 80)
        print("SAMPLE DATA (first 3 records)")
        print("=" * 80)
        sample = conn.execute("""
            SELECT 
                record_id, 
                scan_date, 
                egfr_date, 
                egfr_value, 
                serum_creatinine,
                egfrc,
                date_diff_days,
                arterial_left_kidney_artery,
                arterial_right_kidney_artery
            FROM gold.anon_segmentations_with_egfr 
            LIMIT 3
        """).df()
        print(sample.to_string())
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"✓ Gold schema created")
        print(f"✓ Gold table: gold.anon_segmentations_with_egfr")
        print(f"✓ Total records: {count}")
        print(f"✓ Total columns: {len(result_df.columns)}")
        print(f"✓ Records with eGFR data: {matched}")
        print(f"✓ Records without eGFR data: {unmatched}")
        
        print("\n✓ Data successfully merged to gold layer!")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
