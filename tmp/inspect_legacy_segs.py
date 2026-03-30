import duckdb
import os

def inspect_segments():
    db_path = r'g:\My Drive\kvantify\DanQ_health\analysis\database\egfr_data_v2.duckdb'
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
        
    conn = duckdb.connect(db_path)
    
    print("--- 25-11-2025 (Legacy) Segment Names ---")
    try:
        res = conn.execute("SELECT DISTINCT Segment FROM bronze.legacy_venous_segmentation").df()
        print(res['Segment'].tolist())
    except Exception as e:
        print(f"Error accessing legacy_venous_segmentation: {e}")
        
    print("\n--- 25-11-2025 (data_25112025_segs) Column Names ---")
    try:
        res = conn.execute("DESCRIBE bronze.data_25112025_segs").df()
        print(res['column_name'].tolist())
    except Exception as e:
        print(f"Error accessing data_25112025_segs: {e}")

    conn.close()

if __name__ == "__main__":
    inspect_segments()
