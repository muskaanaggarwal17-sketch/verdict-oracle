"""Build a verified, training-only case-law index from an existing Google Drive cache.

Run from Colab with authenticate() or locally with --local-cache CACHE_DIR.
The deployed Streamlit app DOES NOT need Google credentials, PDFs, or Excel uploads.

This imports excerpts FROM EXISTING EXTRACTED TEXT rather than redownloading PDFs.
PDFs remain in Drive and are only referenced by verified links.
"""
import argparse
import csv
import io
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict,Counter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MANIFEST_ID = '1s2-rOPM7EypqP2mhJmGTw4RaqJhTBATH'
PDF_FOLDER_ID = '1RvRbYXOgZmuutN2LQcdBO_eitIyIYt_E'
TEXT_FOLDER_ID = '19Qu5d2xe8m06Gguhk8OUEStumHc6rbp9'
CACHE_URL = 'https://drive.google.com/drive/folders/1lPVSVbgvyJUheAYqG7gXVN2PILni8T1w'


def legal_name(s):
    s=unicodedata.normalize('NFKC', str(s or '')).casefold()
    s=re.sub(r'\b(?:versus|vs\.?|v\.)\b', 'v', s)
    s=re.sub(r'[^a-z0-9]+', '', s)
    return s


def cite(s):
    return re.sub(r'[^a-z0-9]+', '', unicodedata.normalize('NFKC', str(s or '')).casefold())


def verified_same_case(a,b):
    """Conservative pairwise equality: normalized name AND full citation match.
    Deliberately reject fuzzy/citation-substring matches to avoid wrong precedents.
    """
    return legal_name(a['case_name']) == legal_name(b['case_name']) and cite(a['case_citation']) == cite(b['citation'])


def list_all(service, folder_id):
    items=[]; token=None
    while True:
        d=service.files().list(q=f"'{folder_id}' in parents and trashed=false",
            pageSize=1000,pageToken=token,
            fields='nextPageToken,files(id,name,mimeType,size)',
            supportsAllDrives=True,includeItemsFromAllDrives=True).execute()
        items.extend(d.get('files', []))
        token=d.get('nextPageToken')
        if not token: return items


def media_bytes(service,file_id):
    req=service.files().get_media(fileId=file_id,supportsAllDrives=True)
    stream=io.BytesIO()
    from googleapiclient.http import MediaIoBaseDownload
    downloader=MediaIoBaseDownload(stream,req,chunksize=4*1024*1024)
    done=False
    while not done: _,done=downloader.next_chunk()
    return stream.getvalue()


def authenticated_drive():
    try:
        from google.colab import auth
        auth.authenticate_user()
    except ImportError:
        raise RuntimeError('Run the provided import notebook in Google Colab, or use --local-cache for an already-downloaded cache.')
    import google.auth
    from googleapiclient.discovery import build
    creds,_=google.auth.default()
    return build('drive','v3',credentials=creds,cache_discovery=False)


def assemble(manifest_rows, text_ids, pdf_ids, get_text):
    entries=json.loads((DATA/'training_case_index.json').read_text(encoding='utf-8'))
    # A case must match exactly on both identifiers, after normalizing punctuation/case.
    matches=defaultdict(list)
    for e in entries:
        matches[(legal_name(e['case_name']),cite(e['citation']))].append(e)
    audit=[]; ready=[]; visited=set()
    for row in manifest_rows:
        fid=str(row.get('file_id','')).strip()
        status=str(row.get('status','')).strip().casefold()
        item={'original_file_id':fid,'case_name':row.get('case_name',''),
              'case_citation':row.get('case_citation',''),'status':''}
        if status!='verified':item['status']='source_manifest_not_verified';audit.append(item);continue
        key=(legal_name(row.get('case_name')),cite(row.get('case_citation')))
        candidates=matches.get(key,[])
        if len(candidates)!=1:item['status']='not_unique_training_pair' if candidates else 'not_in_training_split_or_citation_variant';audit.append(item);continue
        if fid in visited:item['status']='duplicate_source_file';audit.append(item);continue
        visited.add(fid)
        if fid not in text_ids or fid not in pdf_ids:item['status']='cached_pdf_or_text_missing';audit.append(item);continue
        try:raw=get_text(fid,text_ids[fid]);txt=raw.decode('utf-8',errors='replace').strip()
        except Exception as ex:item['status']='text_download_error';item['detail']=str(ex)[:180];audit.append(item);continue
        if len(re.sub(r'\s+','',txt))<250:item['status']='extracted_text_too_short';audit.append(item);continue
        e=candidates[0]
        # Limit excerpt to avoid releasing the full judgment and keep web app fast.
        excerpt=re.sub(r'\s+',' ',txt[:13000]).strip()[:6000]
        ready.append({'case_key':e['case_key'],'citation_key':e['citation_key'],
                      'case_name':e['case_name'],'citation':e['citation'],
                      'excerpt':excerpt,'pdf_url':(f'https://drive.google.com/file/d/{pdf_ids[fid]}/view' if pdf_ids[fid] else None),
                      'source_file_id':fid,'match_method':'verified_manifest_exact_name_and_citation',
                      'cached_text_characters':len(txt)})
        item['status']='indexed_verified_training_case';audit.append(item)
    DATA.mkdir(exist_ok=True)
    (DATA/'verified_pdf_excerpts.json').write_text(json.dumps(ready,ensure_ascii=False),encoding='utf-8')
    with open(DATA/'pdf_matching_audit.csv','w',encoding='utf-8',newline='') as fh:
        columns=['original_file_id','case_name','case_citation','status','detail']
        writer=csv.DictWriter(fh,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(audit)
    counts=dict(Counter(x['status'] for x in audit))
    report={'source_cache':CACHE_URL,'verified_manifest_rows':sum(str(x.get('status','')).lower()=='verified' for x in manifest_rows),
        'training_case_records':len(entries),'indexed_training_pdfs':len(ready),
        'matching_status_counts':counts,
        'rule':'Exact normalized case name and full citation; training partition only. No test PDFs indexed.',
        'pdf_bytes_bundled':False,'excerpt_text_bundled':True}
    (DATA/'pdf_coverage_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report


def import_from_drive():
    svc=authenticated_drive()
    manifest=list(csv.DictReader(io.StringIO(media_bytes(svc,MANIFEST_ID).decode('utf-8-sig'))))
    texts={x['name'][:-4]:x['id'] for x in list_all(svc,TEXT_FOLDER_ID) if x['name'].lower().endswith('.txt')}
    pdfs={x['name'][:-4]:x['id'] for x in list_all(svc,PDF_FOLDER_ID) if x['name'].lower().endswith('.pdf')}
    print('Connected cache: manifest',len(manifest),'text records',len(texts),'retained PDFs',len(pdfs))
    return assemble(manifest,texts,pdfs,lambda fid,drive_id:media_bytes(svc,drive_id))


def import_local(folder):
    base=Path(folder)
    with (base/'verified_manifest.csv').open(encoding='utf-8-sig',newline='') as fh:manifest=list(csv.DictReader(fh))
    txts={p.stem:p for p in (base/'extracted_text').glob('*.txt')}
    pdfs={p.stem:p.name for p in (base/'retained_pdfs').glob('*.pdf')}
    # Local mode cannot verify a public Drive PDF copy ID; store original reference only.
    return assemble(manifest,txts,{k:None for k in pdfs},lambda fid,path:path.read_bytes())

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--local-cache',help='Folder with verified_manifest.csv, extracted_text and retained_pdfs')
    args=ap.parse_args()
    import_local(args.local_cache) if args.local_cache else import_from_drive()
