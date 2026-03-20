import pandas as pd
import duckdb
from lib.utils import load_config, setup_logging, get_db_connection

def calculate_egfrc(row):
    """CKD-EPI 2009 formula for eGFRc."""
    try:
        pcc = pd.to_numeric(row.get('serum_creatinine'), errors='coerce')
        age = pd.to_numeric(row.get('current_age'), errors='coerce')
        sex = row.get('sex')
        
        if pd.isna(pcc) or pd.isna(age) or pcc <= 0 or age <= 0:
            return None
            
        # Conversion to mg/dL
        cr = pcc / 88.4
        
        # Gender-specific parameters
        if sex == 'F':
            kappa = 0.7
            alpha = -0.329 if cr <= 0.7 else -1.209
            constant = 144
        elif sex == 'M':
            kappa = 0.9
            alpha = -0.411 if cr <= 0.9 else -1.209
            constant = 141
        else:
            return None
            
        return constant * ((cr / kappa) ** alpha) * (0.993 ** age)
    except:
        return None

def run_silver_layer():
    config = load_config()
    logger = setup_logging("silver_layer", config['paths']['logs'])
    conn = get_db_connection(config['paths']['database'])
    
    silver_schema = config['schemas']['silver']
    bronze_schema = config['schemas']['bronze']

    logger.info("Starting Silver Layer Standardization...")
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {silver_schema}")
    
    # 1. Unify eGFR Data
    logger.info("Unifying eGFR data...")
    # Map sources to standardized columns: record_id, current_age, sex, egfr_date, egfr_value, serum_creatinine, source_folder
    egfr_union_query = f"""
    CREATE OR REPLACE TABLE {silver_schema}.egfr AS
    WITH unified_raw AS (
        -- Source: 25-11-2025 (Join with segs to get demographics)
        SELECT 
            '25-11-2025_' || e.record_id AS record_id, 
            s.current_age, 
            s.sex, 
            e.egfr_date AS egfr_date_str, 
            e.egfr_value, 
            e.serum_creatinine, 
            e.source_folder 
        FROM {bronze_schema}.data_25112025_egfr e
        LEFT JOIN (
            SELECT record_id, current_age, sex FROM {bronze_schema}.data_25112025_segs GROUP BY 1, 2, 3
        ) s ON e.record_id = s.record_id
        
        UNION ALL
        
        -- Source: 12-03-2026
        SELECT 
            '12-03-2026_' || CAST(TRY_CAST(e.record_id AS INTEGER) AS VARCHAR) AS record_id,
            TRY_CAST(REGEXP_EXTRACT(e.age_at_egfr, '~\\s*(\\d+)', 1) AS INTEGER) AS current_age,
            CASE 
                WHEN f.sex = 0 THEN 'F'
                WHEN f.sex = 1 THEN 'M'
                ELSE NULL 
            END AS sex,
            e.egfr_date AS egfr_date_str,
            e.egfr_value,
            e.serum_creatinine,
            e.source_folder
        FROM {bronze_schema}.data_12032026_egfr e
        LEFT JOIN (
            SELECT TRY_CAST(record_id AS INTEGER) as record_id, MAX(sex) as sex FROM {bronze_schema}.data_12032026_full GROUP BY 1
        ) f ON TRY_CAST(e.record_id AS INTEGER) = f.record_id
    )
    SELECT 
        * EXCLUDE (egfr_date_str, sex),
        CASE 
            WHEN LOWER(sex) IN ('m', 'male') THEN 'M'
            WHEN LOWER(sex) IN ('f', 'female') THEN 'F'
            ELSE sex 
        END AS sex,
        COALESCE(
            TRY_CAST(egfr_date_str AS DATE),
            TRY_STRPTIME(egfr_date_str, '%d-%m-%Y'),
            TRY_STRPTIME(egfr_date_str, '%Y-%m-%d'),
            TRY_STRPTIME(egfr_date_str, '%m/%d/%Y')
        ) AS egfr_date
    FROM unified_raw
    """
    try:
        conn.execute(egfr_union_query)
        logger.info("Successfully unified eGFR data with demographics correction and date parsing.")
    except Exception as e:
        logger.error(f"Failed to unify eGFR data: {e}")

    # 2. Add eGFRc to Silver
    df_egfr = conn.execute(f"SELECT * FROM {silver_schema}.egfr").df()
    df_egfr['egfrc'] = df_egfr.apply(calculate_egfrc, axis=1)
    conn.execute(f"CREATE OR REPLACE TABLE {silver_schema}.egfr AS SELECT * FROM df_egfr")
    logger.info("Calculated eGFRc for Silver layer.")

    # 3. Standardize Segmentations
    logger.info("Standardizing Segmentations...")
    
    # 3a. Source: 25-11-2025 (Already Wide)
    df_25112025 = conn.execute(f"SELECT * FROM {bronze_schema}.data_25112025_segs").df()
    df_25112025['record_id'] = '25-11-2025_' + df_25112025['record_id'].astype(str)
    
    # 3b. Source: 12-03-2026 (Long format, needs pivoting and deduplication)
    # First, get the 'best' demographics per record_id (closest to scan_date)
    demog_query = f"""
    SELECT 
        '12-03-2026_' || CAST(TRY_CAST(record_id AS INTEGER) AS VARCHAR) AS record_id,
        MAX(current_age) AS current_age,
        MAX(CASE 
            WHEN sex = 0 THEN 'F'
            WHEN sex = 1 THEN 'M'
            ELSE NULL 
        END) AS sex,
        MAX(scan_date) AS scan_date
    FROM {bronze_schema}.data_12032026_full
    GROUP BY 1
    """
    df_demog_12032026 = conn.execute(demog_query).df()
    
    df_meas_raw = conn.execute(f"SELECT * FROM {bronze_schema}.data_12032026_meas").df()
    
    # Standardize structure names
    def standardize_structure(s):
        s = s.replace('kidney_artery_left', 'left_kidney_artery')
        s = s.replace('kidney_artery_right', 'right_kidney_artery')
        s = s.replace('kidney_vein_left', 'left_kidney_vein')
        s = s.replace('kidney_vein_right', 'right_kidney_vein')
        return s

    df_meas_raw['anatomical_structure'] = df_meas_raw['anatomical_structure'].apply(standardize_structure)
    df_meas_raw['col_name'] = df_meas_raw['phase'] + '_' + df_meas_raw['anatomical_structure']
    
    pivoted_12032026 = df_meas_raw.pivot(index='record_id', columns='col_name', values='iodine_concentration').reset_index()
    pivoted_12032026['record_id'] = '12-03-2026_' + pivoted_12032026['record_id'].astype(int).astype(str)
    pivoted_12032026['source_folder'] = '12-03-2026'
    
    # Merge measurements with the 'best' demographics
    pivoted_12032026 = pivoted_12032026.merge(df_demog_12032026, on='record_id', how='left')
    
    # 4. Union Scans
    final_scans = pd.concat([df_25112025, pivoted_12032026], ignore_index=True, sort=False)
    
    # 5. Filter for demographic and measurement completeness
    metadata_cols = ['record_id', 'current_age', 'sex', 'scan_date', 'source_folder']
    seg_cols = [c for c in final_scans.columns if c not in metadata_cols]
    
    initial_count = len(final_scans)
    # Drop rows with NULL age or sex
    final_scans = final_scans.dropna(subset=['current_age', 'sex'])
    # Drop rows with no measurement data
    final_scans = final_scans.dropna(subset=seg_cols, how='all')
    
    logger.info(f"Filtering: Dropped {initial_count - len(final_scans)} records with missing demographics or measurements.")
    
    # 6. Standardize Sex in segmentations
    final_scans['sex'] = final_scans['sex'].map({
        'male': 'M', 'm': 'M', 'Male': 'M', 'M': 'M',
        'female': 'F', 'f': 'F', 'Female': 'F', 'F': 'F'
    }).fillna(final_scans['sex'])
    
    # Standardize dates in duckdb
    conn.execute(f"CREATE OR REPLACE TABLE {silver_schema}.segs_raw AS SELECT * FROM final_scans")
    conn.execute(f"""
        CREATE OR REPLACE TABLE {silver_schema}.segmentations AS 
        SELECT 
            * EXCLUDE (scan_date),
            COALESCE(
                TRY_CAST(scan_date AS DATE),
                TRY_STRPTIME(scan_date, '%d-%m-%Y'),
                TRY_STRPTIME(scan_date, '%Y-%m-%d'),
                TRY_STRPTIME(scan_date, '%Y/%m/%d')
            ) AS scan_date
        FROM {silver_schema}.segs_raw
    """)
    
    logger.info(f"Silver Layer Completion: Standardized scans and dates.")
    conn.execute(f"DROP TABLE IF EXISTS {silver_schema}.segs_raw")
    conn.close()

if __name__ == "__main__":
    run_silver_layer()
