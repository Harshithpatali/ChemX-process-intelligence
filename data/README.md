# ChemX data policy

Raw research datasets are intentionally excluded from Git.

Use the documented public sources in `data/data_dictionary.md` and the project
download/audit workflow to populate `data/raw/` locally before retraining or
reproducing offline experiments.

The deployed API and Streamlit frontend do **not** require `data/raw/` because they
consume the versioned model bundle and compact derived metadata.

Do not commit raw datasets, archives, credentials, or generated large model artifacts
without an explicit provenance/reproducibility decision.
