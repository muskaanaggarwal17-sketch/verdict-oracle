# Connected Google Drive cache inspection (2026-11-10)

Folder: `1lPVSVbgvyJUheAYqG7gXVN2PILni8T1w`

Observed through connected Drive:
- `verified_manifest.csv` (`1s2-rOPM7EypqP2mhJmGTw4RaqJhTBATH`): 338 `verified` records out of 338 displayed manifest data rows.
- `matching_audit.csv` and `source_inventory.csv` in cache root.
- `retained_pdfs` (`1RvRbYXOgZmuutN2LQcdBO_eitIyIYt_E`): 360 PDF objects.
- `extracted_text` (`19Qu5d2xe8m06Gguhk8OUEStumHc6rbp9`): 360 TXT objects.

The importer makes a *further* exact pair matching to `training_case_index.json` and bundles only those matches, not all 360. Initial project ZIP has no included PDF excerpts. Import once and inspect `pdf_coverage_report.json` before sharing.
