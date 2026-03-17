import duckdb
import pandas as pd

conn = duckdb.connect('g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb')

print('--- gold.master_cases Sample (12-03-2026) ---')
query = "SELECT record_id, current_age, sex, source_folder FROM gold.master_cases WHERE source_folder = '12-03-2026' LIMIT 5"
df = conn.execute(query).df()
print(df.to_string())

print('\n--- Table Existence Check ---')
tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema IN ('silver', 'gold')").df()
print(tables)

conn.close()
