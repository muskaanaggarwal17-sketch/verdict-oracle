# Verdict Oracle — Final shareable web application

This is a **research prototype** based on the original preprocessed Excel dataset, the same trained Logistic Regression and Ridge Regression models, and a verified Google Drive cache containing retained PDF judgments and extracted text. Professor/end users need **only a web URL**, once the owner deploys it.

## Included / independently verified

- `Verdict_Oracle_Preprocessed.xlsx` — the uploaded preprocessed workbook, bundled for reproducibility.
- `models/*.joblib` — pretrained divorce and payment-period-specific maintenance models.
- `data/retained.json`, `data/divorce.json`, `data/maintenance.json` — the same cleaned case data as earlier releases.
- `data/evaluation.json` — group-held-out test accuracy, precision, recall, F1, ROC/AUC, confusion matrix, and maintenance errors.
- `data/training_case_index.json` — **only cases used to fit at least one model**. Neither excluded cases nor held-out test cases are eligible for PDF retrieval.
- `app.py` — interactive web interface, optional facts, explanation, comparable-case law, evaluations and visualisations.
- `import_verified_drive_cache.py` — **one-time owner step** to pull extracted text and verified PDF links from the connected Drive cache (no repeated 1,384-PDF download).
- `Import_Judgments_Once.ipynb` — step-by-step Colab notebook for that one-time import.

The connected cache was inspected: its visible Drive folder contains **360 PDF files and 360 extracted-text files**; **338 entries in `verified_manifest.csv` carry status `verified`**. Those 338 are *candidates*, not 338 guaranteed matches to the model-training partition. Actual coverage is calculated by the import and audit. The ZIP initially does **not** bundle their text; do not claim the PDF stage is complete until the import notebook runs successfully.

## How YOU (project owner) prepare the final deployable ZIP

1. Download `Verdict_Oracle_Final_Application.zip` and **keep it zipped**.
2. Open [Google Colab](https://colab.research.google.com/). Choose **File → Upload notebook** and select `Import_Judgments_Once.ipynb` from the ZIP (extract this one file first).
3. Run all cells. Upload the ZIP when the notebook prompts. Authenticate the Google account you connected to the verified Drive cache. The script accesses these known IDs: manifest `1s2-rOPM7EypqP2mhJmGTw4RaqJhTBATH`, PDF folder `1RvRbYXOgZmuutN2LQcdBO_eitIyIYt_E`, extracted text folder `19Qu5d2xe8m06Gguhk8OUEStumHc6rbp9`.
4. Colab writes `data/verified_pdf_excerpts.json`, `data/pdf_matching_audit.csv`, `data/pdf_coverage_report.json`. Inspect the coverage counts and a few sample citations against their linked PDFs. It then downloads the **completed** `Verdict_Oracle_Deploy_Ready.zip`.
5. Verify case text licensing, privacy and distribution permissions before placing any excerpts or datasets in a public repository. If necessary keep the repository private or omit full-text excerpts.

## How to deploy so your professor only gets a URL

1. Unzip the **completed** `Verdict_Oracle_Deploy_Ready.zip` locally.
2. Create a GitHub repository with its contents. Use a private repository if needed for research confidentiality and ensure your chosen hosting tier supports your repository visibility.
3. Visit [Streamlit Community Cloud](https://share.streamlit.io/), create a new app from that repository, and select `app.py` as the entrypoint. The app automatically installs `requirements.txt`.
4. Wait for the Streamlit deployment to succeed; test a partial-facts input, the five result tabs, the ROC/F1 panel, and a verified PDF link.
5. Send the resulting URL to the professor. **They do not need the ZIP, Excel sheet, Google Drive connection, Colab or Python.**

## Local operation (optional)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How to update models and metrics after changing dataset

```bash
python train_and_evaluate.py
```

This refits models using a grouped 80/20 split. **After retraining, rerun the verified Drive cache import** because the training-only eligible-case index may change. Never load a PDF excerpt from a held-out case into a case-law retrieval index used during evaluation. Leave original source PDFs unchanged.

## Interpret the metrics appropriately

- F1-score, ROC curve and ROC-AUC refer to **binary divorce classification**; they do not evaluate maintenance amounts.
- Maintenance prediction uses MAE and R², calculated only when enough labelled records exist. Period types are modeled separately. Limited labels lead to unstable amounts.
- Model performance is from one grouped holdout split, not cross-jurisdiction validation. Missing inputs use imputation; entering more details may help but does not guarantee accuracy.
- PDF similarity is **post-prediction retrieval**, not model training on PDF content. It identifies potentially useful cases but is not legal validation, a holding extractor, or a legal-reasoning engine.

## Source and limitations

Verified cache: https://drive.google.com/drive/folders/1lPVSVbgvyJUheAYqG7gXVN2PILni8T1w?usp=sharing

The app never needs live Google Drive credentials. Access to linked PDFs depends on the sharing permissions of those individual files. The importer does not move or modify any Drive file. Case-law references should be checked by a human before relying on them. Do not treat output as legal advice.


### Colab authentication fix (November 2026)
Use the notebook included in this release. It authenticates Google Drive in the notebook kernel and imports judgment text in the same process. If importing fails, the full underlying error is displayed rather than only a CalledProcessError.
