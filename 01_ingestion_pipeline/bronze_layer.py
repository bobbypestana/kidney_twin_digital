import os
import pandas as pd
from pathlib import Path
from lib.utils import load_config, setup_logging, get_db_connection

def ingest_31_08_2025(cases_dir, logger):
    """Ingest data from the 31-08-2025 folder (nested case folders)."""
    numbered_folders = [f for f in cases_dir.iterdir() if f.is_dir() and f.name.isdigit()]
    logger.info(f"Source 31-08-2025: Found {len(numbered_folders)} case folders.")
    
    all_egfr = []
    all_segs = []
    
    for folder in numbered_folders:
        # eGFR data
        egfr_file = folder / f"eGFR_{folder.name}.csv"
        if egfr_file.exists():
            df = pd.read_csv(egfr_file, sep=';')
            df['source_folder'] = '31-08-2025'
            df['case_id'] = folder.name
            all_egfr.append(df)
            
        # Segmentations
        seg_dir = folder / "Segmenteringer"
        if seg_dir.exists():
            for phase in ['arterial', 'late', 'venous']:
                phase_file = seg_dir / f"table_{phase}_{folder.name}.csv"
                if phase_file.exists():
                    try:
                        # Check if file is empty
                        if phase_file.stat().st_size == 0:
                            logger.warning(f"File {phase_file} is empty. Skipping.")
                            continue
                            
                        df = pd.read_csv(phase_file) 
                        if df.empty:
                            logger.warning(f"File {phase_file} has no data. Skipping.")
                            continue
                            
                        df['source_folder'] = '31-08-2025'
                        df['case_id'] = folder.name
                        df['phase'] = phase
                        all_segs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to read {phase_file}: {e}. Skipping.")
                    
    return pd.concat(all_egfr) if all_egfr else None, pd.concat(all_segs) if all_segs else None

def ingest_25_11_2025(data_dir, logger):
    """Ingest data from the 25-11-2025 folder (flat CSVs)."""
    anon_egfr = data_dir / "anon_egfr.csv"
    anon_seg = data_dir / "anon_segmentations.csv"
    
    egfr_df = pd.read_csv(anon_egfr, sep=';') if anon_egfr.exists() else None
    seg_df = pd.read_csv(anon_seg, sep=';') if anon_seg.exists() else None
    
    if egfr_df is not None: 
        egfr_df['source_folder'] = '25-11-2025'
        logger.info(f"Source 25-11-2025: Loaded {len(egfr_df)} eGFR rows.")
        
    if seg_df is not None: 
        seg_df['source_folder'] = '25-11-2025'
        logger.info(f"Source 25-11-2025: Loaded {len(seg_df)} segmentation rows.")
        
    return egfr_df, seg_df

def ingest_12_03_2026(data_dir, logger):
    """Ingest data from the 12-03-2026 folder (new flat CSVs)."""
    egfr_file = data_dir / "egfr_measurements.csv"
    vgfr_file = data_dir / "vgfr_measurements.csv"
    
    # These new files use semicolon separators
    egfr_df = pd.read_csv(egfr_file, sep=';') if egfr_file.exists() else None
    vgfr_df = pd.read_csv(vgfr_file, sep=';') if vgfr_file.exists() else None
    
    if egfr_df is not None: 
        egfr_df['source_folder'] = '12-03-2026'
        logger.info(f"Source 12-03-2026: Loaded {len(egfr_df)} eGFR rows.")
        
    if vgfr_df is not None: 
        vgfr_df['source_folder'] = '12-03-2026'
        logger.info(f"Source 12-03-2026: Loaded {len(vgfr_df)} measurement rows.")
        
    return egfr_df, vgfr_df

def run_bronze_layer():
    config = load_config()
    logger = setup_logging("bronze_layer", config['paths']['logs'])
    conn = get_db_connection(config['paths']['database'])
    
    logger.info("Starting Bronze Layer Ingestion...")
    bronze_schema = config['schemas']['bronze']
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {bronze_schema}")
    
    base_path = Path(config['paths']['source_data'])
    
    for source in config['ingestion']['sources']:
        source_name = source['name']
        source_path = base_path / source_name
        
        logger.info(f"Processing source: {source_name}")
        
        if source_name == "31-08-2025":
            egfr_df, seg_df = ingest_31_08_2025(source_path / "Cases", logger)
            if egfr_df is not None:
                conn.execute(f"CREATE OR REPLACE TABLE {bronze_schema}.data_31082025_egfr AS SELECT * FROM egfr_df")
            if seg_df is not None:
                conn.execute(f"CREATE OR REPLACE TABLE {bronze_schema}.data_31082025_segs AS SELECT * FROM seg_df")
                
        elif source_name == "25-11-2025":
            egfr_df, seg_df = ingest_25_11_2025(source_path, logger)
            if egfr_df is not None:
                conn.execute(f"CREATE OR REPLACE TABLE {bronze_schema}.data_25112025_egfr AS SELECT * FROM egfr_df")
            if seg_df is not None:
                conn.execute(f"CREATE OR REPLACE TABLE {bronze_schema}.data_25112025_segs AS SELECT * FROM seg_df")
                
        elif source_name == "12-03-2026":
            egfr_df, meas_df = ingest_12_03_2026(source_path, logger)
            if egfr_df is not None:
                conn.execute(f"CREATE OR REPLACE TABLE {bronze_schema}.data_12032026_egfr AS SELECT * FROM egfr_df")
            if meas_df is not None:
                conn.execute(f"CREATE OR REPLACE TABLE {bronze_schema}.data_12032026_meas AS SELECT * FROM meas_df")
    
    logger.info("Bronze Layer Ingestion Complete.")
    conn.close()

if __name__ == "__main__":
    run_bronze_layer()
