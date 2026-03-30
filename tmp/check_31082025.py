import duckdb
import pandas as pd
from pathlib import Path

db_path = Path(r'g:\My Drive\kvantify\DanQ_health\analysis\database\egfr_data_v2.duckdb')
conn = duckdb.connect(str(db_path), read_only=True)

# Let's check what source folders exist
folders = conn.execute("SELECT DISTINCT source_folder FROM gold.master_cases").fetchall()
print("Source folders in master_cases:", folders)

# If 31-08-2025 exists, let's see its non-null features
df = conn.execute(
    """
    SELECT f.*
    FROM gold.ml_features f
    JOIN gold.master_cases m ON f.record_id = m.record_id
    WHERE m.source_folder = '31-08-2025'
    """
).fetchdf()

if not df.empty:
    X = df.dropna(axis=1, how='all')
    print("\nFeatures that are NOT completely null for 31-08-2025 cohort:")
    cols = sorted([c for c in X.columns if c not in ['record_id', 'egfrc', 'source_folder']])
    print(f"Total feature count: {len(cols)}")
    for c in cols:
        print(f" - {c}")
else:
    print("WARNING: No data found for '31-08-2025' in gold.ml_features!")
    
conn.close()
