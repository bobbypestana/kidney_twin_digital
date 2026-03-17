import argparse
import sys
import importlib
from lib.utils import load_config, setup_logging

# Standard imports now work because the files were renamed
from bronze_layer import run_bronze_layer
from silver_layer import run_silver_layer
from gold_layer import run_gold_layer

def main():
    parser = argparse.ArgumentParser(description="vGFR Data Ingestion & Integration Pipeline")
    parser.add_argument("--layer", choices=["bronze", "silver", "gold", "all"], default="all",
                        help="The pipeline layer to execute.")
    args = parser.parse_all_if_needed() # Placeholder for complex logic, but simple enough for now
    
    config = load_config()
    logger = setup_logging("orchestrator", config['paths']['logs'])
    
    logger.info("="*80)
    logger.info(f"STARTING PIPELINE RUN: {args.layer.upper()}")
    logger.info("="*80)
    
    try:
        if args.layer in ["bronze", "all"]:
            run_bronze_layer()
            
        if args.layer in ["silver", "all"]:
            run_silver_layer()
            
        if args.layer in ["gold", "all"]:
            run_gold_layer()
            
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY.")
        
    except Exception as e:
        logger.error(f"PIPELINE FAILED with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="vGFR Data Ingestion & Integration Pipeline")
    parser.add_argument("--layer", choices=["bronze", "silver", "gold", "all"], default="all",
                        help="The pipeline layer to execute.")
    args = parser.parse_args()
    
    config = load_config()
    logger = setup_logging("orchestrator", config['paths']['logs'])
    
    logger.info("="*80)
    logger.info(f"STARTING PIPELINE RUN: {args.layer.upper()}")
    logger.info("="*80)
    
    try:
        if args.layer in ["bronze", "all"]:
            logger.info(">>> Running Bronze Layer...")
            run_bronze_layer()
            
        if args.layer in ["silver", "all"]:
            logger.info(">>> Running Silver Layer...")
            run_silver_layer()
            
        if args.layer in ["gold", "all"]:
            logger.info(">>> Running Gold Layer...")
            run_gold_layer()
            
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY.")
        
    except Exception as e:
        logger.error(f"PIPELINE FAILED with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
