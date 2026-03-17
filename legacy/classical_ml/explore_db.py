import duckdb
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.max_rows', 30)

conn = duckdb.connect('database/egfr_data.duckdb', read_only=True)

# ===================== GOLD TABLE: main dataset =====================
print("=" * 100)
print("GOLD TABLE: anon_segmentations_with_egfr")
print("=" * 100)

df = conn.execute('SELECT * FROM gold.anon_segmentations_with_egfr').fetchdf()
print(f"Shape: {df.shape}")
print()

# Full data display
print("--- All rows ---")
print(df.to_string())
print()

# Null counts
print("--- Null counts ---")
print(df.isnull().sum().to_string())
print()

# Descriptive stats
print("--- Descriptive statistics ---")
print(df.describe().to_string())
print()

# ===================== BRONZE TABLES =====================
bronze_tables = ['anon_egfr', 'anon_segmentations', 'arterial_segmentation', 'egfr_data', 'late_segmentation', 'venous_segmentation']
for tname in bronze_tables:
    print(f"\n{'=' * 100}")
    print(f"BRONZE TABLE: {tname}")
    print(f"{'=' * 100}")
    
    desc = conn.execute(f'DESCRIBE bronze."{tname}"').fetchdf()
    count = conn.execute(f'SELECT COUNT(*) FROM bronze."{tname}"').fetchone()[0]
    
    print(f"Row count: {count}")
    print(f"\nColumns ({len(desc)}):")
    for _, c in desc.iterrows():
        print(f"  {c['column_name']:40s}  {c['column_type']}")
    
    print("\nSample (3 rows):")
    sample = conn.execute(f'SELECT * FROM bronze."{tname}" LIMIT 3').fetchdf()
    print(sample.to_string())
    print()

conn.close()
