"""
eGFR Data Preparation Script
Converts CSV data from case folders into DuckDB tables.

This script reads:
- eGFR data from case folders
- Segmentation tables (arterial, late, venous)

And stores them in DuckDB tables in the database/ directory.
"""

import duckdb
import pandas as pd
from pathlib import Path
import sys

# Set pandas display options
pd.set_option('display.max_columns', None)

def read_egfr_data(cases_dir):
    """
    Read all eGFR CSV files from numbered case folders.
    
    Args:
        cases_dir: Path to the Cases directory
        
    Returns:
        DataFrame with combined eGFR data
    """
    # Find all numbered subfolders
    numbered_folders = [f for f in cases_dir.iterdir() if f.is_dir() and f.name.isdigit()]
    print(f"Found {len(numbered_folders)} numbered folders: {sorted([f.name for f in numbered_folders], key=int)}")
    
    # Read all eGFR CSV files
    dataframes = []
    for folder in numbered_folders:
        egfr_file = folder / f"eGFR_{folder.name}.csv"
        if egfr_file.exists():
            print(f"Reading {egfr_file}")
            df = pd.read_csv(egfr_file, sep=';')
            df['filename'] = egfr_file.name
            df['case_number'] = folder.name
            dataframes.append(df)
        else:
            print(f"Warning: {egfr_file} not found")
    
    # Combine all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nCombined eGFR dataframe shape: {combined_df.shape}")
        print(f"Columns: {len(combined_df.columns)}")
        print(f"Case numbers included: {sorted(combined_df['case_number'].unique(), key=int)}")
        return combined_df
    else:
        print("No eGFR files found!")
        return None


def read_segmentation_tables(cases_dir, table_type):
    """
    Read all CSV files of a specific table type (arterial, late, venous)
    and combine them into a single dataframe.
    
    Args:
        cases_dir: Path to the Cases directory
        table_type: Type of table ('arterial', 'late', or 'venous')
        
    Returns:
        DataFrame with combined segmentation data
    """
    # Find all numbered subfolders
    numbered_folders = [f for f in cases_dir.iterdir() if f.is_dir() and f.name.isdigit()]
    
    dataframes = []
    
    for folder in numbered_folders:
        segmenteringer_dir = folder / "Segmenteringer"
        if segmenteringer_dir.exists():
            table_file = segmenteringer_dir / f"table_{table_type}_{folder.name}.csv"
            if table_file.exists():
                try:
                    # Check if file is empty or has content
                    file_size = table_file.stat().st_size
                    if file_size == 0:
                        print(f"Warning: {table_file} is empty, skipping")
                        continue
                    
                    print(f"Reading {table_file} ({file_size} bytes)")
                    
                    # Try reading with different parameters to handle various CSV formats
                    try:
                        df = pd.read_csv(table_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(table_file, encoding='latin-1')
                        except:
                            df = pd.read_csv(table_file, encoding='cp1252')
                    
                    # Check if dataframe is empty
                    if df.empty:
                        print(f"Warning: {table_file} contains no data, skipping")
                        continue
                    
                    df['filename'] = table_file.name
                    df['case_number'] = folder.name
                    df['table_type'] = table_type
                    dataframes.append(df)
                    print(f"  Successfully loaded {len(df)} rows, {len(df.columns)} columns")
                    
                except pd.errors.EmptyDataError:
                    print(f"Warning: {table_file} is empty or has no columns, skipping")
                except pd.errors.ParserError as e:
                    print(f"Warning: {table_file} has parsing error: {e}, skipping")
                except Exception as e:
                    print(f"Error reading {table_file}: {e}, skipping")
            else:
                print(f"Warning: {table_file} not found")
        else:
            print(f"Warning: {segmenteringer_dir} not found for case {folder.name}")
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n{table_type.upper()} table combined dataframe shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"Case numbers included: {sorted(combined_df['case_number'].unique(), key=int)}")
        return combined_df
    else:
        print(f"No {table_type} table files found!")
        return None


def main():
    """Main function to read data and create DuckDB tables."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = Path(r'c:\Users\FilipeFurlanBellotti\OneDrive - Kvantify\Kvantify - WP5 - Health Sector Use Case\DanQ-WP5 Kvantify-HGH\data_analisys')
    cases_dir = data_dir / "31-08-2025" / "Cases"
    db_dir = script_dir.parent / "database"
    
    # Create database directory if it doesn't exist
    db_dir.mkdir(exist_ok=True)
    print(f"Database directory: {db_dir}")
    
    # Database file path
    db_file = db_dir / "egfr_data.duckdb"
    print(f"Database file: {db_file}\n")
    
    # Check if cases directory exists
    if not cases_dir.exists():
        print(f"Error: Cases directory not found at {cases_dir}")
        sys.exit(1)
    
    # Read eGFR data
    print("=" * 60)
    print("READING eGFR DATA")
    print("=" * 60)
    egfr_df = read_egfr_data(cases_dir)
    
    # Read segmentation tables
    print("\n" + "=" * 60)
    print("READING ARTERIAL TABLES")
    print("=" * 60)
    arterial_df = read_segmentation_tables(cases_dir, "arterial")
    
    print("\n" + "=" * 60)
    print("READING LATE TABLES")
    print("=" * 60)
    late_df = read_segmentation_tables(cases_dir, "late")
    
    print("\n" + "=" * 60)
    print("READING VENOUS TABLES")
    print("=" * 60)
    venous_df = read_segmentation_tables(cases_dir, "venous")
    
    # Create DuckDB connection and tables
    print("\n" + "=" * 60)
    print("CREATING DUCKDB TABLES")
    print("=" * 60)
    
    conn = duckdb.connect(str(db_file))
    
    try:
        # Create bronze schema
        print("\n" + "=" * 60)
        print("CREATING BRONZE SCHEMA")
        print("=" * 60)
        conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        print("Created schema 'bronze'")
        
        # Create eGFR table
        if egfr_df is not None:
            conn.execute("DROP TABLE IF EXISTS bronze.egfr_data")
            conn.execute("CREATE TABLE bronze.egfr_data AS SELECT * FROM egfr_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.egfr_data").fetchone()[0]
            print(f"Created table 'bronze.egfr_data' with {count} rows")
        
        # Create arterial segmentation table
        if arterial_df is not None:
            conn.execute("DROP TABLE IF EXISTS bronze.arterial_segmentation")
            conn.execute("CREATE TABLE bronze.arterial_segmentation AS SELECT * FROM arterial_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.arterial_segmentation").fetchone()[0]
            print(f"Created table 'bronze.arterial_segmentation' with {count} rows")
        
        # Create late segmentation table
        if late_df is not None:
            conn.execute("DROP TABLE IF EXISTS bronze.late_segmentation")
            conn.execute("CREATE TABLE bronze.late_segmentation AS SELECT * FROM late_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.late_segmentation").fetchone()[0]
            print(f"Created table 'bronze.late_segmentation' with {count} rows")
        
        # Create venous segmentation table
        if venous_df is not None:
            conn.execute("DROP TABLE IF EXISTS bronze.venous_segmentation")
            conn.execute("CREATE TABLE bronze.venous_segmentation AS SELECT * FROM venous_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.venous_segmentation").fetchone()[0]
            print(f"Created table 'bronze.venous_segmentation' with {count} rows")
        
        # Read and create tables from 25-11-2025 directory
        print("\n" + "=" * 60)
        print("READING 25-11-2025 DATA FILES")
        print("=" * 60)
        
        data_2025_dir = data_dir / "25-11-2025"
        
        # Read anon_egfr.csv
        anon_egfr_file = data_2025_dir / "anon_egfr.csv"
        if anon_egfr_file.exists():
            print(f"Reading {anon_egfr_file.name}...")
            anon_egfr_df = pd.read_csv(anon_egfr_file, sep=';')
            conn.execute("DROP TABLE IF EXISTS bronze.anon_egfr")
            conn.execute("CREATE TABLE bronze.anon_egfr AS SELECT * FROM anon_egfr_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.anon_egfr").fetchone()[0]
            print(f"Created table 'bronze.anon_egfr' with {count} rows")
        
        # Read anon_segmentations.csv
        anon_seg_file = data_2025_dir / "anon_segmentations.csv"
        if anon_seg_file.exists():
            print(f"Reading {anon_seg_file.name}...")
            anon_seg_df = pd.read_csv(anon_seg_file, sep=';')
            conn.execute("DROP TABLE IF EXISTS bronze.anon_segmentations")
            conn.execute("CREATE TABLE bronze.anon_segmentations AS SELECT * FROM anon_seg_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.anon_segmentations").fetchone()[0]
            print(f"Created table 'bronze.anon_segmentations' with {count} rows")
        
        # Read iodine_concentration_table.csv
        iodine_file = data_2025_dir / "iodine_concentration_table.csv"
        if iodine_file.exists():
            print(f"Reading {iodine_file.name}...")
            iodine_df = pd.read_csv(iodine_file)
            conn.execute("DROP TABLE IF EXISTS bronze.iodine_concentration")
            conn.execute("CREATE TABLE bronze.iodine_concentration AS SELECT * FROM iodine_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.iodine_concentration").fetchone()[0]
            print(f"Created table 'bronze.iodine_concentration' with {count} rows")
        
        # Read case_by_case_comparison.csv
        comparison_file = data_2025_dir / "case_by_case_comparison.csv"
        if comparison_file.exists():
            print(f"Reading {comparison_file.name}...")
            comparison_df = pd.read_csv(comparison_file)
            conn.execute("DROP TABLE IF EXISTS bronze.case_comparison")
            conn.execute("CREATE TABLE bronze.case_comparison AS SELECT * FROM comparison_df")
            count = conn.execute("SELECT COUNT(*) FROM bronze.case_comparison").fetchone()[0]
            print(f"Created table 'bronze.case_comparison' with {count} rows")
        
        # Show all tables
        print("\n" + "=" * 60)
        print("DATABASE SUMMARY")
        print("=" * 60)
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables in database: {[t[0] for t in tables]}")
        
        print("\nData successfully loaded into DuckDB!")
        print(f"Database location: {db_file}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
