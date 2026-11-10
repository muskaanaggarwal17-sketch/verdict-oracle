"""OPTIONAL one-time PDF excerpt indexing. Run on the owner's computer or Colab.
Usage: python build_pdf_index.py /path/to/retained_pdf_folder
No Google Drive connection is needed after committing verified_pdf_excerpts.json.
Only training cases in data/training_case_index.json are eligible.
"""
import sys,json,re,unicodedata
from pathlib import Path
import fitz
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
def norm(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKC',str(s)).casefold())
def main(folder):
    entries=json.loads((DATA/'training_case_index.json').read_text(encoding='utf-8'))
    source=Path(folder)
    if not source.is_dir():raise FileNotFoundError(source)
    pdfs=list(source.rglob('*.pdf'))
    # Avoid fuzzy matching: only unique citation matches from PDF text are accepted.
    matches={};audit=[]
    for i,p in enumerate(pdfs,1):
        try:
            with fitz.open(p) as doc:
                text='\n'.join(page.get_text() for page in list(doc)[:min(len(doc),8)])
        except Exception as exc:
            audit.append({'pdf':p.name,'status':'unreadable','reason':str(exc)[:120]});continue
        clean=norm(text[:50000])
        if len(clean)<250:
            audit.append({'pdf':p.name,'status':'scanned_or_insufficient_text'});continue
        candidates=[e for e in entries if len(norm(e['citation']))>=9 and norm(e['citation']) in clean]
        if len(candidates)!=1:
            audit.append({'pdf':p.name,'status':'ambiguous' if candidates else 'unmatched','candidate_count':len(candidates)});continue
        e=candidates[0];k=(e['case_key'],e['citation_key'])
        matches.setdefault(k,[]).append({'case_key':e['case_key'],'citation_key':e['citation_key'],
             'excerpt':re.sub(r'\s+',' ',text[:8500]).strip(),'source_filename':p.name})
        audit.append({'pdf':p.name,'status':'candidate_match','citation':e['citation']})
        if i%100==0:print('Inspected',i,'/',len(pdfs))
    verified=[values[0] for values in matches.values() if len(values)==1]
    (DATA/'verified_pdf_excerpts.json').write_text(json.dumps(verified,ensure_ascii=False),encoding='utf-8')
    (DATA/'pdf_matching_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified unique training-case PDF excerpts:',len(verified),'of',len(entries),'training records')
    print('Unmatched or ambiguous files remain excluded. Scanned PDFs require a separate OCR and verification process.')
if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python build_pdf_index.py /path/to/retained_pdf_folder')
    main(sys.argv[1])
