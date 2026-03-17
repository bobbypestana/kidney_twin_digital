import pandas as pd
from pathlib import Path

file_path = Path(r"c:\Users\FilipeFurlanBellotti\OneDrive - Kvantify\Kvantify - WP5 - Health Sector Use Case\DanQ-WP5 Kvantify-HGH\data_analisys\12-03-2026\egfr_measurements.csv")

print(f"File size: {file_path.stat().st_size}")

try:
    df = pd.read_csv(file_path, sep=';')
    print("\nColumns (sep=';'):")
    print(df.columns.tolist())
    print("\nFirst 2 rows:")
    print(df.head(2).to_string())
except Exception as e:
    print(f"Error with sep=';': {e}")

try:
    df2 = pd.read_csv(file_path)
    print("\nColumns (default sep=','):")
    print(df2.columns.tolist())
except Exception as e:
    print(f"Error with sep=',': {e}")
