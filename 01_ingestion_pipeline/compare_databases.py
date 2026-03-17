import duckdb
import pandas as pd

# Paths
LEGACY_DB = 'g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data.duckdb'
NEW_DB = 'g:/My Drive/kvantify/DanQ_health/analysis/database/egfr_data_v2.duckdb'

def main():
    try:
        # Load New Data (25 patients)
        new_conn = duckdb.connect(NEW_DB)
        new_df = new_conn.execute("SELECT record_id, current_age, sex, egfrc, arterial_left_kidney_artery FROM gold.master_cases").df()
        new_conn.close()
        new_df['record_id'] = new_df['record_id'].astype(str)
        
        # Load Legacy Data
        # We need to use a temp file or a different connection string if read_only fails
        legacy_conn = duckdb.connect(LEGACY_DB, read_only=True)
        legacy_df = legacy_conn.execute("SELECT record_id, current_age, sex, egfrc, arterial_left_kidney_artery FROM gold.anon_segmentations_with_egfr").df()
        legacy_conn.close()
        legacy_df['record_id'] = legacy_df['record_id'].astype(str)
        
        # Merge on record_id
        merged = pd.merge(new_df, legacy_df, on='record_id', suffixes=('_new', '_legacy'))
        
        print(f"Merged {len(merged)} overlapping records.")
        
        if len(merged) > 0:
            print("\n--- Feature Comparison ---")
            for col in ['current_age', 'egfrc', 'arterial_left_kidney_artery']:
                diff = (merged[f'{col}_new'] - merged[f'{col}_legacy']).abs().mean()
                print(f"Mean Abs Diff for {col}: {diff:.4f}")
            
            print("\nSample records:")
            print(merged[['record_id', 'egfrc_new', 'egfrc_legacy', 'current_age_new', 'current_age_legacy']].head())
        else:
            print("\nNO OVERLAP FOUND! Check record_id formats.")
            print("New record_ids sample:", new_df['record_id'].tolist()[:5])
            print("Legacy record_ids sample:", legacy_df['record_id'].tolist()[:5])

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
