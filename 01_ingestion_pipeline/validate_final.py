import duckdb
import pandas as pd

database_path = 'g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb'
conn = duckdb.connect(database_path)

print("PIPELINE DATA SUMMARY")
print("="*40)

for schema in ['bronze', 'silver', 'gold']:
    print(f"\n[{schema.upper()} LAYER]")
    try:
        tables = conn.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'").fetchall()
        for (t_name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {schema}.{t_name}").fetchone()[0]
            print(f"  {schema}.{t_name:25s}: {count:6d} rows")
    except Exception as e:
        print(f"  Error reading schema {schema}: {e}")

conn.close()
