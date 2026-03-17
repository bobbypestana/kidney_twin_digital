"""
Create Gold Layer Segmentation Table

This script transforms bronze layer segmentation data (arterial, venous, late)
from long format to wide format, creating a gold layer table with one row per case.

Only kidney-related measurements (renal artery and vein for left/right) are included.
"""

import duckdb
import pandas as pd
from pathlib import Path

def main():
    """Main function to create gold layer segmentation table."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent
    db_file = data_dir / "database" / "egfr_data.duckdb"
    
    print(f"Database file: {db_file}\n")
    
    # Connect to database
    conn = duckdb.connect(str(db_file))
    
    try:
        # Create gold schema
        print("=" * 60)
        print("CREATING GOLD SCHEMA")
        print("=" * 60)
        conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
        print("Created schema 'gold'\n")
        
        # Read bronze segmentation tables
        print("=" * 60)
        print("READING BRONZE SEGMENTATION TABLES")
        print("=" * 60)
        
        arterial_df = conn.execute('SELECT * FROM bronze.arterial_segmentation').df()
        venous_df = conn.execute('SELECT * FROM bronze.venous_segmentation').df()
        late_df = conn.execute('SELECT * FROM bronze.late_segmentation').df()
        
        print(f"Arterial: {len(arterial_df)} rows")
        print(f"Venous: {len(venous_df)} rows")
        print(f"Late: {len(late_df)} rows\n")
        
        # Transform arterial data
        print("=" * 60)
        print("TRANSFORMING ARTERIAL DATA")
        print("=" * 60)
        
        arterial_kidney = arterial_df[
            arterial_df['Segment'].str.contains('renal_artery|renal_vein', case=False, na=False)
        ].copy()
        
        # Create column names based on segment
        def create_column_name(segment):
            """Map segment name to gold column name."""
            segment_lower = segment.lower()
            
            if 'renal_artery_left' in segment_lower:
                return 'arterial_left_kidney_artery'
            elif 'renal_artery_right' in segment_lower:
                return 'arterial_right_kidney_artery'
            elif 'renal_vein_left' in segment_lower:
                return 'arterial_left_kidney_vein'
            elif 'renal_vein_right' in segment_lower:
                return 'arterial_right_kidney_vein'
            return None
        
        arterial_kidney['column_name'] = arterial_kidney['Segment'].apply(create_column_name)
        arterial_kidney = arterial_kidney[arterial_kidney['column_name'].notna()]
        
        arterial_pivot = arterial_kidney.pivot(
            index='case_number',
            columns='column_name',
            values='Mean'
        ).reset_index()
        
        print(f"Arterial pivoted: {arterial_pivot.shape}")
        print(f"Columns: {list(arterial_pivot.columns)}\n")
        
        # Transform venous data
        print("=" * 60)
        print("TRANSFORMING VENOUS DATA")
        print("=" * 60)
        
        venous_kidney = venous_df[
            venous_df['Segment'].str.contains('renal_artery|renal_vein', case=False, na=False)
        ].copy()
        
        def create_venous_column_name(segment):
            """Map segment name to gold column name for venous."""
            segment_lower = segment.lower()
            
            if 'renal_artery_left' in segment_lower:
                return 'venous_left_kidney_artery'
            elif 'renal_artery_right' in segment_lower:
                return 'venous_right_kidney_artery'
            elif 'renal_vein_left' in segment_lower:
                return 'venous_left_kidney_vein'
            elif 'renal_vein_right' in segment_lower:
                return 'venous_right_kidney_vein'
            return None
        
        venous_kidney['column_name'] = venous_kidney['Segment'].apply(create_venous_column_name)
        venous_kidney = venous_kidney[venous_kidney['column_name'].notna()]
        
        venous_pivot = venous_kidney.pivot(
            index='case_number',
            columns='column_name',
            values='Mean'
        ).reset_index()
        
        print(f"Venous pivoted: {venous_pivot.shape}")
        print(f"Columns: {list(venous_pivot.columns)}\n")
        
        # Transform late data
        print("=" * 60)
        print("TRANSFORMING LATE DATA")
        print("=" * 60)
        
        late_kidney = late_df[
            late_df['Segment'].str.contains('renal_artery|renal_vein', case=False, na=False)
        ].copy()
        
        def create_late_column_name(segment):
            """Map segment name to gold column name for late."""
            segment_lower = segment.lower()
            
            if 'renal_artery_left' in segment_lower:
                return 'late_left_kidney_artery'
            elif 'renal_artery_right' in segment_lower:
                return 'late_right_kidney_artery'
            elif 'renal_vein_left' in segment_lower:
                return 'late_left_kidney_vein'
            elif 'renal_vein_right' in segment_lower:
                return 'late_right_kidney_vein'
            return None
        
        late_kidney['column_name'] = late_kidney['Segment'].apply(create_late_column_name)
        late_kidney = late_kidney[late_kidney['column_name'].notna()]
        
        late_pivot = late_kidney.pivot(
            index='case_number',
            columns='column_name',
            values='Mean'
        ).reset_index()
        
        print(f"Late pivoted: {late_pivot.shape}")
        print(f"Columns: {list(late_pivot.columns)}\n")
        
        # Merge all phases
        print("=" * 60)
        print("MERGING ALL PHASES")
        print("=" * 60)
        
        gold_segmentations = arterial_pivot.merge(
            venous_pivot, on='case_number', how='outer'
        ).merge(
            late_pivot, on='case_number', how='outer'
        )
        
        print(f"Gold segmentations shape: {gold_segmentations.shape}")
        print(f"Columns: {list(gold_segmentations.columns)}")
        print(f"Cases: {sorted(gold_segmentations['case_number'].unique(), key=int)}\n")
        
        # Create gold table
        print("=" * 60)
        print("CREATING GOLD TABLE")
        print("=" * 60)
        
        conn.execute("DROP TABLE IF EXISTS gold.segmentations")
        conn.execute("CREATE TABLE gold.segmentations AS SELECT * FROM gold_segmentations")
        
        count = conn.execute("SELECT COUNT(*) FROM gold.segmentations").fetchone()[0]
        print(f"Created table 'gold.segmentations' with {count} rows")
        
        # Show sample
        print("\nSample data (case 3):")
        sample = conn.execute("SELECT * FROM gold.segmentations WHERE case_number = '3'").df()
        print(sample.T)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Gold schema created: ✓")
        print(f"Gold table: gold.segmentations")
        print(f"Total cases: {count}")
        print(f"Total columns: {len(gold_segmentations.columns)}")
        print(f"Kidney measurements per case: {len(gold_segmentations.columns) - 1}")  # -1 for case_number
        
        print("\nData successfully transformed to gold layer!")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
