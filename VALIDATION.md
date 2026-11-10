# Validation and limitations

- App and importer Python syntax checked in the creation environment.
- The pretrained `divorce.joblib` was loaded and predicted successfully on a single partially completed structured case (ground for divorce only).
- The importer was exercised using a locally fabricated verified manifest, extracted text and a training-case match; it produced one verified case entry and an audit.
- Structured training files include `retained.json`, `divorce.json`, `maintenance.json` and `training_case_index.json`.
- The Streamlit GUI itself and full authenticated cache import have **not** been executed end-to-end here. They require deployment and owner-side Drive authorization respectively.
- The Google Drive folders and their metadata were inspected through the newly connected Google Drive access. 360 PDFs and 360 text files were listed. The 338 verified-manifest rows have **not** been fully transferred into this ZIP because content distribution and match-to-training verification are part of the one-time import.

## Existing grouped holdout metrics

- Divorce accuracy: **0.7165**
- Divorce macro F1: **0.7125**
- Divorce ROC–AUC: **0.7639**
- Divorce held-out test cases: **127**

These values are from a single grouped 80/20 split on the saved structured data, not live validation or prospectively held-out cases. Maintenance samples are small; inspect `data/evaluation.json` for MAE/R² and sample sizes.
