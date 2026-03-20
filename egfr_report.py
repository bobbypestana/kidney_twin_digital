import duckdb
import pandas as pd

db_path = 'g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb'
conn = duckdb.connect(db_path, read_only=True)

query = """
WITH analysis AS (
    SELECT 
        source_folder,
        TRY_CAST(egfr_value AS DOUBLE) as egfr_v,
        egfrc,
        CASE 
            WHEN TRY_CAST(egfr_value AS DOUBLE) <= 90 THEN (ABS(TRY_CAST(egfr_value AS DOUBLE) - egfrc) > 1)
            WHEN TRY_CAST(egfr_value AS DOUBLE) > 90 THEN (egfrc <= 90)
            ELSE FALSE
        END AS is_different
    FROM silver.egfr
    WHERE egfrc IS NOT NULL AND TRY_CAST(egfr_value AS DOUBLE) IS NOT NULL
)
SELECT 
    source_folder,
    COUNT(*) as total_records,
    SUM(CASE WHEN is_different THEN 1 ELSE 0 END) as different_records,
    ROUND(SUM(CASE WHEN is_different THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_different
FROM analysis
GROUP BY 1
ORDER BY 4 DESC
"""

df = conn.execute(query).df()
print(df.to_string(index=False))
conn.close()
