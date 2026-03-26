"""Bronze layer ingestion pipeline.

Reads raw source CSVs into the DuckDB bronze schema. No transformations —
data is stored as-is for full traceability.
"""
import logging
import re
import pandas as pd
from pathlib import Path
from typing import Optional
from lib.utils import load_config, setup_logging, get_db_connection


# ---------------------------------------------------------------------------
# Slicer Data Column Names (CSV and XLSX use the same schema)
# ---------------------------------------------------------------------------
_SLICER_COLS = [
    "Segment",
    "Voxel count (LM)", "Volume mm3 (LM)", "Volume cm3 (LM)",
    "Voxel count (SV)", "Volume mm3 (SV)", "Volume cm3 (SV)",
    "Minimum", "Maximum", "Mean", "Standard deviation",
    "Percentile 5", "Percentile 95", "Median",
]


# ---------------------------------------------------------------------------
# 31-08-2025 (nested case folders)
# ---------------------------------------------------------------------------

def ingest_31_08_2025(
    cases_dir: Path,
    logger: logging.Logger,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Ingest data from the 31-08-2025 folder (nested case folders).

    Args:
        cases_dir: Root path containing one numbered subfolder per case.
        logger: Configured logger instance.

    Returns:
        Tuple of (egfr_df, segmentations_df). Either may be None if no data
        was found.
    """
    numbered_folders = [f for f in cases_dir.iterdir() if f.is_dir() and f.name.isdigit()]
    logger.info(f"Source 31-08-2025: Found {len(numbered_folders)} case folders.")

    all_egfr: list[pd.DataFrame] = []
    all_segs: list[pd.DataFrame] = []

    for folder in numbered_folders:
        egfr_file = folder / f"eGFR_{folder.name}.csv"
        if egfr_file.exists():
            df = pd.read_csv(egfr_file, sep=";")
            df["source_folder"] = "31-08-2025"
            df["case_id"] = folder.name
            all_egfr.append(df)

        seg_dir = folder / "Segmenteringer"
        if seg_dir.exists():
            for phase in ["arterial", "late", "venous"]:
                phase_file = seg_dir / f"table_{phase}_{folder.name}.csv"
                if phase_file.exists():
                    try:
                        if phase_file.stat().st_size == 0:
                            logger.warning(f"File {phase_file} is empty. Skipping.")
                            continue
                        df = pd.read_csv(phase_file)
                        if df.empty:
                            logger.warning(f"File {phase_file} has no data. Skipping.")
                            continue
                        df["source_folder"] = "31-08-2025"
                        df["case_id"] = folder.name
                        df["phase"] = phase
                        all_segs.append(df)
                    except Exception as exc:
                        logger.warning(f"Failed to read {phase_file}: {exc}. Skipping.")

    return (
        pd.concat(all_egfr, ignore_index=True) if all_egfr else None,
        pd.concat(all_segs, ignore_index=True) if all_segs else None,
    )


# ---------------------------------------------------------------------------
# 25-11-2025 (flat CSVs)
# ---------------------------------------------------------------------------

def ingest_25_11_2025(
    data_dir: Path,
    logger: logging.Logger,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Ingest data from the 25-11-2025 folder (flat CSVs).

    Args:
        data_dir: Directory containing ``anon_egfr.csv`` and
            ``anon_segmentations.csv``.
        logger: Configured logger instance.

    Returns:
        Tuple of (egfr_df, segmentations_df). Either may be None if the
        corresponding file is absent.
    """
    anon_egfr = data_dir / "anon_egfr.csv"
    anon_seg = data_dir / "anon_segmentations.csv"

    egfr_df = pd.read_csv(anon_egfr, sep=";") if anon_egfr.exists() else None
    seg_df = pd.read_csv(anon_seg, sep=";") if anon_seg.exists() else None

    if egfr_df is not None:
        egfr_df["source_folder"] = "25-11-2025"
        logger.info(f"Source 25-11-2025: Loaded {len(egfr_df)} eGFR rows.")

    if seg_df is not None:
        seg_df["source_folder"] = "25-11-2025"
        logger.info(f"Source 25-11-2025: Loaded {len(seg_df)} segmentation rows.")

    return egfr_df, seg_df


# ---------------------------------------------------------------------------
# 12-03-2026 – flat CSVs
# ---------------------------------------------------------------------------

def ingest_12_03_2026(
    data_dir: Path,
    logger: logging.Logger,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Ingest the flat CSV files from the 12-03-2026 folder.

    Args:
        data_dir: Directory containing ``egfr_measurements.csv``,
            ``vgfr_measurements.csv``, and ``Combined_full_dataset.csv``.
        logger: Configured logger instance.

    Returns:
        Tuple of (egfr_df, vgfr_df, full_df). Any element may be None if the
        corresponding file is absent.
    """
    egfr_file = data_dir / "egfr_measurements.csv"
    vgfr_file = data_dir / "vgfr_measurements.csv"
    full_file = data_dir / "Combined_full_dataset.csv"

    egfr_df = pd.read_csv(egfr_file, sep=";") if egfr_file.exists() else None
    vgfr_df = pd.read_csv(vgfr_file, sep=";") if vgfr_file.exists() else None
    full_df = pd.read_csv(full_file, sep=";") if full_file.exists() else None

    if egfr_df is not None:
        egfr_df["source_folder"] = "12-03-2026"
        logger.info(f"Source 12-03-2026: Loaded {len(egfr_df)} eGFR rows.")

    if vgfr_df is not None:
        vgfr_df["source_folder"] = "12-03-2026"
        logger.info(f"Source 12-03-2026: Loaded {len(vgfr_df)} vGFR measurement rows.")

    if full_df is not None:
        full_df["source_folder"] = "12-03-2026"
        logger.info(f"Source 12-03-2026: Loaded {len(full_df)} combined dataset rows.")

    return egfr_df, vgfr_df, full_df


# ---------------------------------------------------------------------------
# 12-03-2026 – Raw slicer data (CT segmentation statistics per case/phase)
# ---------------------------------------------------------------------------

def _read_slicer_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Read a single slicer table file (CSV or XLSX).

    The file format has one row per anatomical segment with HU statistics.
    The segment name encodes ``{record_id}_{phase}_{structure}``.

    Args:
        filepath: Path to a ``.csv`` or ``.xlsx`` slicer table.

    Returns:
        A normalised DataFrame with columns
        [record_id, phase, structure, volume_cm3, hu_mean, hu_std,
        hu_median, source_folder], or None on error.

    Raises:
        None — errors are caught and returned as None.
    """
    try:
        if filepath.suffix == ".xlsx":
            df = pd.read_excel(filepath, engine="openpyxl")
            # XLSX: first column is the segment identifier
            df.columns = [c.strip() for c in df.columns]
            seg_col = df.columns[0]
            df = df.rename(columns={seg_col: "Segment"})
        else:
            # CSV: header row may be wrapped in one outer double-quote (pathological)
            # or be a standard quoted CSV. We detect this by inspecting the header.
            import io
            text = filepath.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if not lines:
                return None
            
            first_line = lines[0].strip()
            # Pathological: starts/ends with quote and has ," (standard is ",")
            is_pathological = (
                first_line.startswith('"') and 
                first_line.endswith('"') and 
                ',""' in first_line
            )

            if is_pathological:
                # Clean each line by stripping outer quotes and reducing doubled inner quotes
                cleaned_lines = []
                for l in lines:
                    l = l.strip()
                    if l.startswith('"') and l.endswith('"'):
                        l = l[1:-1]
                    l = l.replace('""', '"')
                    cleaned_lines.append(l)
                df = pd.read_csv(io.StringIO("\n".join(cleaned_lines)))
            else:
                df = pd.read_csv(io.StringIO(text))

            df.columns = [c.strip().strip('"') for c in df.columns]

        # Keep only the columns we care about
        keep = {
            "Segment": "Segment",
            "Volume cm3 (LM)": "volume_cm3",
            "Mean": "hu_mean",
            "Standard deviation": "hu_std",
            "Median": "hu_median",
        }
        df = df[[c for c in keep if c in df.columns]].rename(columns=keep)

        # Coerce numeric columns: handle both dot and comma decimal separators.
        numeric_cols = ["volume_cm3", "hu_mean", "hu_std", "hu_median"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )


        # Parse record_id, phase, structure out of "Segment" values.
        # Pattern: {record_id}_{phase}_{structure}, e.g. 109_arterial_aorta
        stem = filepath.stem  # e.g. "109_arterial_table"
        m = re.match(r"^(\d+)_(arterial|venous|late)_table$", stem)
        if not m:
            return None
        record_id = int(m.group(1))
        phase = m.group(2)

        # Strip "{record_id}_{phase}_" prefix from Segment to get structure name.
        # Normalize by stripping whitespace and replacing spaces with underscores
        # so that XLSX entries like "hepatic vein_right" map to the same name
        # as CSV entries "hepatic_vein_right".
        prefix = f"{record_id}_{phase}_"
        df["structure"] = (
            df["Segment"]
            .str.replace(prefix, "", regex=False)
            .str.strip()
            .str.replace(" ", "_", regex=False)
        )
        df["record_id"] = record_id
        df["phase"] = phase
        df["source_folder"] = "12-03-2026"

        return df[["record_id", "phase", "structure", "volume_cm3", "hu_mean", "hu_std", "hu_median", "source_folder"]]

    except Exception as exc:
        return None


def ingest_raw_slicer_data(
    slicer_dir: Path,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Ingest CT segmentation statistics from the *Raw slicer data* folder.

    Scans for files matching ``{record_id}_{phase}_table.{csv,xlsx}``.
    Both formats are supported transparently. When a record has both, the
    CSV is preferred (it is extracted from the XLSX by the analyst).

    Args:
        slicer_dir: Path to ``12-03-2026/Raw slicer data``.
        logger: Configured logger instance.

    Returns:
        Concatenated DataFrame with one row per (record_id, phase,
        structure), or None if the directory is absent or empty.
    """
    if not slicer_dir.exists():
        logger.warning(f"Slicer data directory not found: {slicer_dir}")
        return None

    # Collect all candidate files; deduplicate by stem, preferring CSV
    files_by_stem: dict[str, Path] = {}
    pattern = re.compile(r"^\d+_(arterial|venous|late)_table\.(csv|xlsx)$")
    for f in slicer_dir.iterdir():
        if not pattern.match(f.name):
            continue
        stem = f.stem
        if stem not in files_by_stem:
            files_by_stem[stem] = f
        elif f.suffix == ".csv":
            # Prefer CSV over XLSX for the same stem
            files_by_stem[stem] = f

    logger.info(
        f"Source 12-03-2026/Raw slicer data: Found {len(files_by_stem)} table files."
    )

    frames: list[pd.DataFrame] = []
    skipped = 0
    for stem, filepath in sorted(files_by_stem.items()):
        result = _read_slicer_file(filepath)
        if result is not None:
            frames.append(result)
        else:
            skipped += 1
            logger.warning(f"Skipped unreadable slicer file: {filepath.name}")

    if not frames:
        logger.warning("No valid slicer data files were parsed.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Source 12-03-2026/Raw slicer data: Ingested {len(combined)} rows "
        f"from {len(frames)} files ({skipped} skipped)."
    )
    return combined


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_bronze_layer() -> None:
    """Run the full bronze-layer ingestion for all configured sources.

    Raises:
        RuntimeError: If the DuckDB connection cannot be established.
    """
    config = load_config()
    logger = setup_logging("bronze_layer", config["paths"]["logs"])
    conn = get_db_connection(config["paths"]["database"])

    logger.info("Starting Bronze Layer Ingestion...")
    bronze_schema = config["schemas"]["bronze"]
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {bronze_schema}")

    base_path = Path(config["paths"]["source_data"])

    for source in config["ingestion"]["sources"]:
        source_name: str = source["name"]
        source_path = base_path / source_name
        logger.info(f"Processing source: {source_name}")

        if source_name == "31-08-2025":
            egfr_df, seg_df = ingest_31_08_2025(source_path / "Cases", logger)
            if egfr_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_31082025_egfr "
                    "AS SELECT * FROM egfr_df"
                )
            if seg_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_31082025_segs "
                    "AS SELECT * FROM seg_df"
                )

        elif source_name == "25-11-2025":
            egfr_df, seg_df = ingest_25_11_2025(source_path, logger)
            if egfr_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_25112025_egfr "
                    "AS SELECT * FROM egfr_df"
                )
            if seg_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_25112025_segs "
                    "AS SELECT * FROM seg_df"
                )

        elif source_name == "12-03-2026":
            egfr_df, vgfr_df, full_df = ingest_12_03_2026(source_path, logger)
            if egfr_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_12032026_egfr "
                    "AS SELECT * FROM egfr_df"
                )
            if vgfr_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_12032026_meas "
                    "AS SELECT * FROM vgfr_df"
                )
            if full_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_12032026_full "
                    "AS SELECT * FROM full_df"
                )

            # NEW: Ingest Raw slicer data (CT segmentation statistics)
            slicer_dir = source_path / "Raw slicer data"
            slicer_df = ingest_raw_slicer_data(slicer_dir, logger)
            if slicer_df is not None:
                conn.execute(
                    f"CREATE OR REPLACE TABLE {bronze_schema}.data_12032026_slicer "
                    "AS SELECT * FROM slicer_df"
                )

    logger.info("Bronze Layer Ingestion Complete.")
    conn.close()


if __name__ == "__main__":
    run_bronze_layer()
