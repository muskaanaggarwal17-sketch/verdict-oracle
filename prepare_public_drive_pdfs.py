"""ONE-TIME owner-side preparation. Public Drive download, then verify citations against training cases.
Run locally or in Colab: python prepare_public_drive_pdfs.py
If gdown cannot list a large folder, download the folder with Google Drive UI and run:
    python prepare_public_drive_pdfs.py --local-folder /path/to/downloaded/folder
The deployed app never calls Drive. This script does not alter trained models.
"""
from pathlib import Path
import argparse, subprocess, sys, json
ROOT=Path(__file__).resolve().parent
FOLDER_URL="https://drive.google.com/drive/folders/1lPVSVbgvyJUheAYqG7gXVN2PILni8T1w?usp=sharing"
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--local-folder',default=None,help='Existing downloaded Drive folder; skips downloading')
    p.add_argument('--download-dir',default=str(ROOT/'owner_pdf_downloads'))
    args=p.parse_args()
    if args.local_folder:
        folder=Path(args.local_folder).expanduser().resolve()
    else:
        try:import gdown
        except ImportError:raise SystemExit('Install first: pip install gdown')
        folder=Path(args.download_dir).resolve();folder.mkdir(parents=True,exist_ok=True)
        print('Downloading publicly shared folder. This is a one-time owner operation.')
        try:
            downloaded=gdown.download_folder(url=FOLDER_URL,output=str(folder),quiet=False,remaining_ok=True,use_cookies=False)
        except Exception as e:
            raise SystemExit(f'Drive download failed: {e}\nFor large folders or quota restrictions, download through Drive and use --local-folder.')
        if not downloaded:raise SystemExit('No files downloaded. Check folder public permissions, or use --local-folder.')
    if not folder.is_dir():raise SystemExit(f'Folder does not exist: {folder}')
    pdfs=list(folder.rglob('*.pdf'))+list(folder.rglob('*.PDF'))
    print('PDFs found:',len(pdfs))
    if not pdfs:raise SystemExit('No PDFs discovered. Check downloaded folder or its nested folders.')
    subprocess.run([sys.executable,str(ROOT/'build_pdf_index.py'),str(folder)],check=True)
    audit=json.loads((ROOT/'data'/'pdf_matching_audit.json').read_text(encoding='utf-8'))
    verified=json.loads((ROOT/'data'/'verified_pdf_excerpts.json').read_text(encoding='utf-8'))
    report={'source_url':FOLDER_URL,'pdfs_discovered':len(pdfs),'verified_training_pdf_excerpts':len(verified),
            'matching_status_counts':{s:sum(x.get('status')==s for x in audit) for s in sorted(set(x.get('status') for x in audit))},
            'important':'Only unique citation matches to model-training records are indexed. Unmatched and ambiguous PDFs excluded.'}
    (ROOT/'data'/'pdf_coverage_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Now deploy app.py together with data/verified_pdf_excerpts.json and data/pdf_coverage_report.json.')
if __name__=='__main__':main()
