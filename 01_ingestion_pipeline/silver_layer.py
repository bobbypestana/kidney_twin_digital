"""Silver layer standardization pipeline.

Reads from the bronze schema and produces clean, joined, wide-format tables
in the silver schema, ready for gold-layer feature engineering.
"""
import logging
import pandas as pd
import duckdb
from typing import Optional
from lib.utils import load_config, setup_logging, get_db_connection


# ---------------------------------------------------------------------------
# eGFRc Calculation (CKD-EPI 2009)
# ---------------------------------------------------------------------------

def calculate_egfrc(row: pd.Series) -> Optional[float]:
    """Calculate eGFRc using the CKD-EPI 2009 formula.

    Args:
        row: A pandas Series containing at least 'serum_creatinine',
            'current_age', and 'sex' fields.

    Returns:
        eGFRc in ml/min/1.73 m², or None if inputs are invalid or missing.
    """
    try:
        pcc = pd.to_numeric(row.get("serum_creatinine"), errors="coerce")
        age = pd.to_numeric(row.get("current_age"), errors="coerce")
        sex = row.get("sex")

        if pd.isna(pcc) or pd.isna(age) or pcc <= 0 or age <= 0:
            return None

        cr = pcc / 88.4  # Convert µmol/L → mg/dL

        if sex == "F":
            kappa, alpha, constant = 0.7, (-0.329 if cr <= 0.7 else -1.209), 144
        elif sex == "M":
            kappa, alpha, constant = 0.9, (-0.411 if cr <= 0.9 else -1.209), 141
        else:
            return None

        return constant * ((cr / kappa) ** alpha) * (0.993**age)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_silver_layer() -> None:
    """Run the full silver-layer standardization pipeline.

    Reads from all bronze sources, standardises schemas (record_id
    prefix, sex encoding, date formats), calculates eGFRc, pivots
    the new Raw Slicer Data, and writes unified tables:

    * ``silver.egfr``          — unified eGFR records with eGFRc
    * ``silver.segmentations`` — unified wide-format scan measurements

    Raises:
        Exception: Propagated from DuckDB if schema creation fails.
    """
    config = load_config()
    logger = setup_logging("silver_layer", config["paths"]["logs"])
    conn = get_db_connection(config["paths"]["database"])

    silver_schema = config["schemas"]["silver"]
    bronze_schema = config["schemas"]["bronze"]

    logger.info("Starting Silver Layer Standardization...")
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {silver_schema}")

    # ------------------------------------------------------------------
    # 1. Unify eGFR data from all sources
    # ------------------------------------------------------------------
    logger.info("Unifying eGFR data...")
    egfr_union_query = f"""
    CREATE OR REPLACE TABLE {silver_schema}.egfr AS
    WITH unified_raw AS (
        -- Source: 25-11-2025 (Join with segs to get demographics)
        SELECT
            '25-11-2025_' || e.record_id                              AS record_id,
            s.current_age,
            s.sex,
            e.egfr_date                                               AS egfr_date_str,
            e.egfr_value,
            e.serum_creatinine,
            e.source_folder
        FROM {bronze_schema}.data_25112025_egfr e
        LEFT JOIN (
            SELECT record_id, current_age, sex
            FROM {bronze_schema}.data_25112025_segs
            GROUP BY 1, 2, 3
        ) s ON e.record_id = s.record_id

        UNION ALL

        -- Source: 12-03-2026 (age extracted inline; sex from full dataset)
        SELECT
            '12-03-2026_' || CAST(TRY_CAST(e.record_id AS INTEGER) AS VARCHAR) AS record_id,
            TRY_CAST(REGEXP_EXTRACT(e.age_at_egfr, '~\\s*(\\d+)', 1) AS INTEGER) AS current_age,
            CASE WHEN f.sex = 0 THEN 'F'
                 WHEN f.sex = 1 THEN 'M'
                 ELSE NULL END                                        AS sex,
            e.egfr_date                                               AS egfr_date_str,
            e.egfr_value,
            e.serum_creatinine,
            e.source_folder
        FROM {bronze_schema}.data_12032026_egfr e
        LEFT JOIN (
            SELECT TRY_CAST(record_id AS INTEGER) AS record_id,
                   MAX(sex) AS sex
            FROM {bronze_schema}.data_12032026_full
            GROUP BY 1
        ) f ON TRY_CAST(e.record_id AS INTEGER) = f.record_id
    )
    SELECT
        * EXCLUDE (egfr_date_str, sex),
        CASE
            WHEN LOWER(sex) IN ('m', 'male')   THEN 'M'
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
        logger.info("Unified eGFR data with demographics and date parsing.")
    except Exception as exc:
        logger.error(f"Failed to unify eGFR data: {exc}")

    # ------------------------------------------------------------------
    # 2. Calculate eGFRc (CKD-EPI 2009) via Python UDF
    # ------------------------------------------------------------------
    df_egfr = conn.execute(f"SELECT * FROM {silver_schema}.egfr").df()
    df_egfr["egfrc"] = df_egfr.apply(calculate_egfrc, axis=1)
    conn.execute(f"CREATE OR REPLACE TABLE {silver_schema}.egfr AS SELECT * FROM df_egfr")
    logger.info("Calculated eGFRc for Silver layer.")

    # ------------------------------------------------------------------
    # 3a. Standardise 25-11-2025 segmentations (already wide)
    # ------------------------------------------------------------------
    logger.info("Standardizing Segmentations...")
    df_25112025 = conn.execute(f"SELECT * FROM {bronze_schema}.data_25112025_segs").df()
    df_25112025["record_id"] = "25-11-2025_" + df_25112025["record_id"].astype(str)

    # ------------------------------------------------------------------
    # 3b. Pivot 12-03-2026 vGFR measurements (long → wide)
    # ------------------------------------------------------------------
    demog_query = f"""
    SELECT
        '12-03-2026_' || CAST(TRY_CAST(record_id AS INTEGER) AS VARCHAR) AS record_id,
        MAX(current_age) AS current_age,
        MAX(CASE WHEN sex = 0 THEN 'F'
                 WHEN sex = 1 THEN 'M'
                 ELSE NULL END)                                           AS sex,
        MAX(scan_date)   AS scan_date
    FROM {bronze_schema}.data_12032026_full
    GROUP BY 1
    """
    df_demog = conn.execute(demog_query).df()

    df_meas_raw = conn.execute(f"SELECT * FROM {bronze_schema}.data_12032026_meas").df()

    def _standardize_structure(s: str) -> str:
        """Normalise anatomical structure names to a common convention."""
        return (
            s.replace("kidney_artery_left", "left_kidney_artery")
             .replace("kidney_artery_right", "right_kidney_artery")
             .replace("kidney_vein_left", "left_kidney_vein")
             .replace("kidney_vein_right", "right_kidney_vein")
        )

    df_meas_raw["anatomical_structure"] = df_meas_raw["anatomical_structure"].apply(
        _standardize_structure
    )
    df_meas_raw["col_name"] = df_meas_raw["phase"] + "_" + df_meas_raw["anatomical_structure"]

    pivoted_12032026 = (
        df_meas_raw
        .pivot(index="record_id", columns="col_name", values="iodine_concentration")
        .reset_index()
    )
    pivoted_12032026["record_id"] = "12-03-2026_" + pivoted_12032026["record_id"].astype(int).astype(str)
    pivoted_12032026["source_folder"] = "12-03-2026"

    # Merge with demographics
    pivoted_12032026 = pivoted_12032026.merge(df_demog, on="record_id", how="left")

    # ------------------------------------------------------------------
    # 3c. Pivot 12-03-2026 Raw Slicer Data — vascular structure HU stats
    # ------------------------------------------------------------------
    # 'threshold' and 'margin' are explicitly excluded (inconsistently
    # labelled whole-organ masks). All other structures provide per-phase
    # HU mean and std which are pivoted to wide-format columns:
    # e.g. arterial_aorta_hu_mean, venous_renal_artery_right_hu_std, ...
    slicer_raw = conn.execute("""
        SELECT
            '12-03-2026_' || CAST(record_id AS VARCHAR) AS record_id,
            phase || '_' || structure || '_hu_mean' AS col_name,
            hu_mean AS val
        FROM bronze.data_12032026_slicer
        WHERE structure NOT IN ('threshold', 'margin')

        UNION ALL

        SELECT
            '12-03-2026_' || CAST(record_id AS VARCHAR) AS record_id,
            phase || '_' || structure || '_hu_std' AS col_name,
            hu_std AS val
        FROM bronze.data_12032026_slicer
        WHERE structure NOT IN ('threshold', 'margin')
    """).df()

    if not slicer_raw.empty:
        df_slicer_wide = (
            slicer_raw
            .pivot_table(index="record_id", columns="col_name", values="val", aggfunc="first")
            .reset_index()
        )
        df_slicer_wide.columns.name = None
        pivoted_12032026 = pivoted_12032026.merge(df_slicer_wide, on="record_id", how="left")
        n_with_slicer = df_slicer_wide["record_id"].nunique()
        logger.info(f"Slicer HU stats joined for {n_with_slicer} records.")

    # ------------------------------------------------------------------
    # 4. Union all sources
    # ------------------------------------------------------------------
    final_scans = pd.concat([df_25112025, pivoted_12032026], ignore_index=True, sort=False)

    # ------------------------------------------------------------------
    # 5. Filter for demographic and measurement completeness
    # ------------------------------------------------------------------
    metadata_cols = ["record_id", "current_age", "sex", "scan_date", "source_folder"]
    seg_cols = [c for c in final_scans.columns if c not in metadata_cols]

    initial_count = len(final_scans)
    final_scans = final_scans.dropna(subset=["current_age", "sex"])
    final_scans = final_scans.dropna(subset=seg_cols, how="all")
    logger.info(
        f"Filtering: Dropped {initial_count - len(final_scans)} records "
        "with missing demographics or measurements."
    )

    # ------------------------------------------------------------------
    # 6. Standardise sex encoding
    # ------------------------------------------------------------------
    sex_map = {
        "male": "M", "m": "M", "Male": "M", "M": "M",
        "female": "F", "f": "F", "Female": "F", "F": "F",
    }
    final_scans["sex"] = final_scans["sex"].map(sex_map).fillna(final_scans["sex"])

    # ------------------------------------------------------------------
    # 7. Write segmentations to silver (with date normalisation)
    # ------------------------------------------------------------------
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

    logger.info("Silver Layer Complete: Standardised scans, dates, and slicer features.")
    conn.execute(f"DROP TABLE IF EXISTS {silver_schema}.segs_raw")
    conn.close()


if __name__ == "__main__":
    run_silver_layer()
