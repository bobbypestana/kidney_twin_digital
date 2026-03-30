import subprocess
import sys

commands = [
    # Legacy cohort (25-11-2025)
    ["conda", "run", "-n", "nrrd-viewer", "python", "02_ml_pipeline/01_repro_r2_r10.py", "--cohort", "25-11-2025", "--exclude-vol-hu"],
    ["conda", "run", "-n", "nrrd-viewer", "python", "02_ml_pipeline/02_improvements_v3.py", "--cohort", "25-11-2025", "--exclude-vol-hu"],
    ["conda", "run", "-n", "nrrd-viewer", "python", "02_ml_pipeline/03_single_phase_experiment.py", "--cohort", "25-11-2025"],
    
    # New cohort (12-03-2026)
    ["conda", "run", "-n", "nrrd-viewer", "python", "02_ml_pipeline/01_repro_r2_r10.py", "--cohort", "12-03-2026"],
    ["conda", "run", "-n", "nrrd-viewer", "python", "02_ml_pipeline/02_improvements_v3.py", "--cohort", "12-03-2026"],
    ["conda", "run", "-n", "nrrd-viewer", "python", "02_ml_pipeline/03_single_phase_experiment.py", "--cohort", "12-03-2026"],
]

for cmd in commands:
    print(f"\n=======================================================\n>>> Running: {' '.join(cmd)}\n=======================================================")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error executing {' '.join(cmd)}!")
        sys.exit(1)
        
print("\nAll scripts executed successfully!")
