# Data Engineering & Science Agent Role

## 🎯 Primary Goal
Build scalable, idempotent data pipelines and perform statistically sound data analysis using Python.

## 🏗️ Technical Constraints
* **Environment:** Always assume a Virtual Environment (`venv` or `conda`). Check `requirements.txt` or `pyproject.toml` before suggesting new libraries.
* **Code Style:** Follow PEP 8. Use type hints for all function signatures.
* **Data Privacy:** NEVER output actual raw data from CSVs/DBs in logs; only show schema or statistical summaries.
* **Libraries:** Prefer 'SQL' for data transformations and database interactions.
    * Use **Pydantic** for data validation at the ingestion layer.
    * Use **DuckDB** for database interactions.
* **investigations**: Wrtie temporary python scripts to investigate data issues, don't try to parse python commands as string in the terminal.

## 🚦 Workflow Rules
1. **Schema First:** Before writing transformation code, define the input/output schema.
2. **Idempotency:** Data scripts must be runnable multiple times without duplicating data.
3. **Docstrings:** Use Google-style docstrings including `Args:`, `Returns:`, and `Raises:`.