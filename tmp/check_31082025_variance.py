import duckdb
from pathlib import Path

db_path = Path(r'g:\My Drive\kvantify\DanQ_health\analysis\database\egfr_data_v2.duckdb')
conn = duckdb.connect(str(db_path), read_only=True)

print("Checking bronze.data_31082025_segs columns:")
try:
    cols = conn.execute("DESCRIBE bronze.data_31082025_segs").fetchall()
    for col in cols:
        print(f" - {col[0]}")
        
    print("\nSample Data:")
    df = conn.execute("SELECT * FROM bronze.data_31082025_segs LIMIT 1").fetchdf()
    print(df.to_string())
except duckdb.CatalogException:
    print("Table bronze.data_31082025_segs does not exist!")

print("\nChecking bronze.data_25112025_segs columns:")
try:
    cols = conn.execute("DESCRIBE bronze.data_25112025_segs").fetchall()
    for col in cols:
        print(f" - {col[0]}")
except duckdb.CatalogException:
    print("Table bronze.data_25112025_segs does not exist!")
    
conn.close()
