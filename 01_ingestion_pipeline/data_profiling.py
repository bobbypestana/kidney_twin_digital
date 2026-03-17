import duckdb
import pandas as pd

conn = duckdb.connect('g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb')

print('--- Bronze Counts ---')
# 25-11-2025
b_segs = conn.execute("SELECT COUNT(*) FROM bronze.data_25112025_segs").fetchone()[0]
b_egfr = conn.execute("SELECT COUNT(*) FROM bronze.data_25112025_egfr").fetchone()[0]
print(f"25-11-2025: Bronze Segs={b_segs}, Bronze eGFR={b_egfr}")

# 12-03-2026
b_segs_new = conn.execute("SELECT COUNT(*) FROM bronze.data_12032026_meas").fetchone()[0]
b_egfr_new = conn.execute("SELECT COUNT(*) FROM bronze.data_12032026_egfr").fetchone()[0]
print(f"12-03-2026: Bronze Segs={b_segs_new}, Bronze eGFR={b_egfr_new}")

print('\n--- Silver Counts ---')
# Segmentations in Silver
s_segs = conn.execute("SELECT source_folder, COUNT(*) FROM silver.segmentations GROUP BY 1").df()
print("Silver Segmentations by source:")
print(s_segs)

# eGFR in Silver
s_egfr = conn.execute("SELECT source_folder, COUNT(*) FROM silver.egfr GROUP BY 1").df()
print("\nSilver eGFR by source:")
print(s_egfr)

print('\n--- Gold Counts ---')
g_master = conn.execute("SELECT source_folder, COUNT(*) FROM gold.master_cases GROUP BY 1").df()
print("Gold Master Cases by source:")
print(g_master)

# Check record_id overlap for 25-11-2025
overlap = conn.execute("""
    SELECT COUNT(DISTINCT s.record_id) 
    FROM silver.segmentations s
    JOIN silver.egfr e ON s.record_id = e.record_id
    WHERE s.source_folder = '25-11-2025'
""").fetchone()[0]
print(f"\nDistict record_id overlap in Silver (25-11-2025): {overlap}")

conn.close()
