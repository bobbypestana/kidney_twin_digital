import json
import glob
from pathlib import Path

# Path to new cohort JSON summaries
json_files = glob.glob(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline\ml_results\12032026\*_summary.json')

for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        try:
            print(f"\n--- {data['round']} / {data['model']} ---")
            features = data['features']
            for feat in features:
                # Is it an HU or Vol segment from Slicer?
                if '_hu_' in feat or '_vol' in feat:
                    origin = "NEW DATASET (Slicer Segments)"
                else:
                    origin = "FIRST DATASET (Legacy Point Measurements / Demographics)"
                print(f"- {feat} -> {origin}")
        except KeyError:
            continue
