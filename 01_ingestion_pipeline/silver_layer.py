import pandas as pd
import duckdb
from lib.utils import load_config, setup_logging, get_db_connection

def calculate_egfrc(row):
    """CKD-EPI formula for eGFRc."""
    try:
        pcc = pd.to_numeric(row.get('serum_creatinine'), errors='coerce')
        age = pd.to_numeric(row.get('current_age'), errors='coerce')
        
        if pd.isna(pcc) or pd.isna(age) or pcc <= 0 or age <= 0:
            return None
            
        alpha = -0.411 if pcc <= 80 else -1.209
        return 141 * ((pcc / (0.9 * 88.4)) ** alpha) * (0.993 ** age)
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
            e.record_id, 
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
    
    # 3a. Source: 25-11-2025 (Already Wide, just mapping)
    pivoted_25112025 = conn.execute(f"SELECT * FROM {bronze_schema}.data_25112025_segs").df()
    pivoted_25112025['record_id'] = pivoted_25112025['record_id'].astype(str)
    
    # Standardize Sex
    if 'sex' in pivoted_25112025.columns:
        pivoted_25112025['sex'] = pivoted_25112025['sex'].map({
            'male': 'M', 'm': 'M', 'Male': 'M',
            'female': 'F', 'f': 'F', 'Female': 'F', 'F': 'F'
        }).fillna(pivoted_25112025['sex'])
    
    # 4. Union Scans
    final_scans = pd.concat([pivoted_25112025], ignore_index=True, sort=False)
    # Standardize dates in duckdb
    conn.execute(f"CREATE OR REPLACE TABLE {silver_schema}.segs_raw AS SELECT * FROM final_scans")
    conn.execute(f"""
        CREATE OR REPLACE TABLE {silver_schema}.segmentations AS 
        SELECT 
            * EXCLUDE (scan_date),
            COALESCE(
                TRY_CAST(scan_date AS DATE),
                TRY_STRPTIME(scan_date, '%d-%m-%Y'),
                TRY_STRPTIME(scan_date, '%Y-%m-%d')
            ) AS scan_date
        FROM {silver_schema}.segs_raw
    """)
    
    logger.info(f"Silver Layer Completion: Standardized scans and dates.")
    conn.execute(f"DROP TABLE IF EXISTS {silver_schema}.egfr_raw")
    conn.execute(f"DROP TABLE IF EXISTS {silver_schema}.segs_raw")
    conn.close()

if __name__ == "__main__":
    run_silver_layer()
