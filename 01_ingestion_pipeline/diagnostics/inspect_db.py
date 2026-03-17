import duckdb
import pandas as pd

conn = duckdb.connect('g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb')

for t_name in ['data_31082025_egfr', 'data_25112025_egfr', 'data_12032026_egfr']:
    print(f"\n--- bronze.{t_name} ---")
    try:
        desc = conn.execute(f"DESCRIBE bronze.{t_name}").df()
        print(desc[['column_name']].to_string())
    except Exception as e:
        print(f"Error: {e}")

conn.close()
