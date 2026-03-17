import pandas as pd
import duckdb
from lib.utils import load_config, setup_logging, get_db_connection

def run_gold_layer():
    config = load_config()
    logger = setup_logging("gold_layer", config['paths']['logs'])
    conn = get_db_connection(config['paths']['database'])
    
    gold_schema = config['schemas']['gold']
    silver_schema = config['schemas']['silver']
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {gold_schema}")
    
    logger.info("Starting Gold Layer Synthesis (Closest eGFR Join)...")
    
    # Use SQL to find the closest eGFR for each scan
    closest_query = f"""
    CREATE OR REPLACE TABLE {gold_schema}.master_cases AS
    WITH ranked_egfr AS (
        SELECT 
            s.* EXCLUDE (current_age, sex),
            COALESCE(e.current_age, s.current_age) AS current_age,
            COALESCE(e.sex, s.sex) AS sex,
            e.egfr_date,
            e.egfr_value,
            e.serum_creatinine,
            e.egfrc,
            ABS(DATEDIFF('day', 
                TRY_CAST(s.scan_date AS DATE), 
                TRY_CAST(e.egfr_date AS DATE)
            )) AS date_diff_days,
            ROW_NUMBER() OVER (
                PARTITION BY s.record_id, s.scan_date 
                ORDER BY ABS(DATEDIFF('day', 
                    TRY_CAST(s.scan_date AS DATE), 
                    TRY_CAST(e.egfr_date AS DATE)
                )) ASC
            ) as rn
        FROM {silver_schema}.segmentations s
        LEFT JOIN {silver_schema}.egfr e ON s.record_id = e.record_id
    )
    SELECT * EXCLUDE (rn)
    FROM ranked_egfr
    WHERE (rn = 1 OR rn IS NULL)
      AND egfr_date IS NOT NULL
    """
    
    try:
        conn.execute(closest_query)
        
        # 1. Pivot Legacy Bronze Features
        logger.info("Pivoting Legacy Bronze Features...")
        pivot_sqls = []
        for phase in ['arterial', 'venous', 'late']:
            pivot_sqls.append(f"""
                SELECT 
                    CAST(case_number AS VARCHAR) as record_id,
                    MAX(CASE WHEN Segment LIKE '%threshold%' THEN "Volume cm3 (LM)" END) as {phase}_kidney_vol,
                    MAX(CASE WHEN Segment LIKE '%threshold%' THEN "Mean" END) as {phase}_kidney_hu_mean,
                    MAX(CASE WHEN Segment LIKE '%threshold%' THEN "Standard deviation" END) as {phase}_kidney_hu_std,
                    MAX(CASE WHEN Segment LIKE '%threshold%' THEN "Median" END) as {phase}_kidney_hu_median
                FROM bronze.legacy_{phase}_segmentation
                GROUP BY 1
            """)
        
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE bronze_pivoted AS
            SELECT 
                a.record_id,
                a.arterial_kidney_vol, a.arterial_kidney_hu_mean, a.arterial_kidney_hu_std, a.arterial_kidney_hu_median,
                v.venous_kidney_vol, v.venous_kidney_hu_mean, v.venous_kidney_hu_std, v.venous_kidney_hu_median,
                l.late_kidney_vol, l.late_kidney_hu_mean, l.late_kidney_hu_std, l.late_kidney_hu_median
            FROM ({pivot_sqls[0]}) a
            LEFT JOIN ({pivot_sqls[1]}) v ON a.record_id = v.record_id
            LEFT JOIN ({pivot_sqls[2]}) l ON a.record_id = l.record_id
        """)

        # 2. Main ML Feature Engineering
        logger.info("Creating Gold ML Features table with Non-Linear Interactions...")
        ml_features_query = f"""
        CREATE OR REPLACE TABLE {gold_schema}.ml_features AS
        WITH base AS (
            SELECT m.*, b.* EXCLUDE (record_id) 
            FROM {gold_schema}.master_cases m
            LEFT JOIN bronze_pivoted b ON m.record_id = b.record_id
        ),
        calculated_ratios AS (
            SELECT 
                *,
                (arterial_left_kidney_artery - arterial_left_kidney_vein) / NULLIF(arterial_left_kidney_artery, 0) AS E_arterial_left,
                (arterial_right_kidney_artery - arterial_right_kidney_vein) / NULLIF(arterial_right_kidney_artery, 0) AS E_arterial_right,
                (venous_left_kidney_artery - venous_left_kidney_vein) / NULLIF(venous_left_kidney_artery, 0) AS E_venous_left,
                (venous_right_kidney_artery - venous_right_kidney_vein) / NULLIF(venous_right_kidney_artery, 0) AS E_venous_right,
                (late_left_kidney_artery - late_left_kidney_vein) / NULLIF(late_left_kidney_artery, 0) AS E_late_left,
                (late_right_kidney_artery - late_right_kidney_vein) / NULLIF(late_right_kidney_artery, 0) AS E_late_right,
                
                arterial_left_kidney_artery / NULLIF(arterial_aorta, 0) AS norm_art_artery_left,
                arterial_right_kidney_artery / NULLIF(arterial_aorta, 0) AS norm_art_artery_right,
                venous_left_kidney_artery / NULLIF(venous_aorta, 0) AS norm_ven_artery_left,
                venous_right_kidney_artery / NULLIF(venous_aorta, 0) AS norm_ven_artery_right
            FROM base
        ),
        mean_excretions AS (
            SELECT 
                *,
                (E_arterial_left + E_arterial_right) / 2 AS E_arterial_mean,
                (E_venous_left + E_venous_right) / 2 AS E_venous_mean,
                (E_late_left + E_late_right) / 2 AS E_late_mean,
                
                (arterial_left_kidney_artery + arterial_right_kidney_artery) / 2 AS mean_artery_arterial,
                (venous_left_kidney_artery + venous_right_kidney_artery) / 2 AS mean_artery_venous
            FROM calculated_ratios
        ),
        interactions AS (
            SELECT 
                *,
                current_age * E_arterial_mean AS age_x_E_arterial,
                current_age * arterial_kidney_vol AS age_x_vol,
                E_arterial_mean * mean_artery_arterial AS art_flow_efficiency,
                E_venous_mean * mean_artery_venous AS ven_flow_efficiency,
                arterial_kidney_vol / NULLIF(current_age, 0) AS vol_per_age
            FROM mean_excretions
        )
        SELECT * FROM interactions
        """
        conn.execute(ml_features_query)
        
        count = conn.execute(f"SELECT COUNT(*) FROM {gold_schema}.ml_features").fetchone()[0]
        logger.info(f"Gold Layer Synthesis Complete: Generated ml_features ({count} records) with {len(conn.execute('DESCRIBE gold.ml_features').df())} columns.")
    except Exception as e:
        logger.error(f"Failed to synthesize Gold layer: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        conn.close()

if __name__ == "__main__":
    run_gold_layer()
