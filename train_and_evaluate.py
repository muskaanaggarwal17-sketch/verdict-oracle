"""One-time, reproducible training/evaluation from the supplied preprocessed workbook export.
Run before deployment. Deployment itself never reads Excel or Google Drive.
"""
from pathlib import Path
import json, re, unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,
                             f1_score,roc_auc_score,roc_curve,confusion_matrix,mean_absolute_error,
                             median_absolute_error,r2_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'models';OUT.mkdir(exist_ok=True)
CATS=['Type of Matrimonial Proceeding','Ground of Divorce','Employment Status of Husband','Employment Status of Wife']
NUMS=['Marriage Duration Years','Separation Duration Years','Children Count','Husband Income INR','Wife Income INR']
FEATURES=CATS+NUMS
PERIODS=['monthly','lump_sum_or_settlement','annual']

def key(s):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s or '')).casefold()).strip()

def read(name):
    return pd.DataFrame(json.loads((DATA/name).read_text(encoding='utf-8')))

def prepare(d):
    d=d.copy()
    for c in CATS:
        if c not in d:d[c]='Unknown'
        d[c]=d[c].fillna('Unknown').astype(str).replace({'':'Unknown'})
    for c in NUMS:
        if c not in d:d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['case_key']=d['Case Name'].map(key)
    d['citation_key']=d['Case Citation'].map(key)
    d=d[d.citation_key.ne('')].drop_duplicates(['case_key','citation_key']).copy()
    return d

def pipeline(kind):
    pre=ColumnTransformer([
        ('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='Unknown')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore'))]),CATS),
        ('num',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value=-1,keep_empty_features=True,add_indicator=False)),
                         ('scale',StandardScaler())]),NUMS)])
    estimator=LogisticRegression(max_iter=2500,class_weight='balanced',random_state=42) if kind=='divorce' else Ridge(alpha=10)
    return Pipeline([('prep',pre),('model',estimator)])

def main():
    divorce=prepare(read('divorce.json'));maintenance=prepare(read('maintenance.json'))
    retained=prepare(read('retained.json'))
    # Global group split for all tasks: no caption appears in both train and test.
    groups=retained.case_key.where(retained.case_key.ne(''),retained.citation_key)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    train_i,test_i=next(splitter.split(retained,groups=groups))
    train_groups=set(groups.iloc[train_i]);test_groups=set(groups.iloc[test_i]);assert not train_groups&test_groups
    def split(d):
        g=d.case_key.where(d.case_key.ne(''),d.citation_key)
        return d[g.isin(train_groups)].copy(),d[g.isin(test_groups)].copy()
    report={'method':'80/20 grouped holdout, fixed seed 42; case captions grouped across tasks',
            'source':'Verdict_Oracle_Preprocessed.xlsx, retained/divorce/maintenance sheets',
            'retained_rows':len(retained),'divorce_labeled_rows':len(divorce),'maintenance_labeled_rows':len(maintenance),
            'models':{}}
    used=set()
    tr,te=split(divorce)
    tr=tr[tr['Divorce Label'].isin(['granted','refused'])];te=te[te['Divorce Label'].isin(['granted','refused'])]
    if len(tr)>10 and len(te)>3 and tr['Divorce Label'].nunique()==2:
        m=pipeline('divorce');m.fit(tr[FEATURES],tr['Divorce Label'])
        joblib.dump(m,OUT/'divorce.joblib',compress=3)
        pred=m.predict(te[FEATURES]);p=m.predict_proba(te[FEATURES])[:,list(m.classes_).index('granted')]
        fpr,tpr,thresholds=roc_curve((te['Divorce Label']=='granted').astype(int),p)
        pr,re,fs,sup=precision_recall_fscore_support(te['Divorce Label'],pred,labels=['granted','refused'],zero_division=0)
        report['models']['divorce']={'status':'trained','train_n':len(tr),'test_n':len(te),
          'accuracy':float(accuracy_score(te['Divorce Label'],pred)),
          'balanced_accuracy':float(balanced_accuracy_score(te['Divorce Label'],pred)),
          'macro_f1':float(f1_score(te['Divorce Label'],pred,average='macro',zero_division=0)),
          'weighted_f1':float(f1_score(te['Divorce Label'],pred,average='weighted',zero_division=0)),
          'roc_auc':float(roc_auc_score((te['Divorce Label']=='granted').astype(int),p)) if te['Divorce Label'].nunique()==2 else None,
          'roc':{'fpr':fpr.tolist(),'tpr':tpr.tolist()},
          'confusion':confusion_matrix(te['Divorce Label'],pred,labels=['granted','refused']).tolist(),
          'classes':['granted','refused'],
          'per_class':{label:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(fs[i]),'support':int(sup[i])} for i,label in enumerate(['granted','refused'])}}
        used.update(zip(tr.case_key,tr.citation_key))
    else:report['models']['divorce']={'status':'insufficient split/class diversity'}
    for period in PERIODS:
        d=maintenance[maintenance['Award Periodicity'].eq(period)].copy()
        d['Amount Awarded INR']=pd.to_numeric(d['Amount Awarded INR'],errors='coerce')
        d=d[d['Amount Awarded INR'].notna()&(d['Amount Awarded INR']>=0)]
        tr,te=split(d)
        name='maintenance_'+period
        if len(tr)<12 or len(te)<3:
            report['models'][name]={'status':'insufficient labeled examples','train_n':len(tr),'test_n':len(te)}
            continue
        m=pipeline('maintenance');m.fit(tr[FEATURES],np.log1p(tr['Amount Awarded INR']))
        joblib.dump(m,OUT/(name+'.joblib'),compress=3)
        y=te['Amount Awarded INR'].to_numpy(dtype=float)
        pred=np.maximum(0,np.expm1(np.clip(m.predict(te[FEATURES]),-10,20)))
        report['models'][name]={'status':'trained','train_n':len(tr),'test_n':len(te),
          'mae_inr':float(mean_absolute_error(y,pred)),
          'median_ae_inr':float(median_absolute_error(y,pred)),
          'r2':float(r2_score(y,pred)) if len(y)>1 else None,
          'median_training_award_inr':float(tr['Amount Awarded INR'].median())}
        used.update(zip(tr.case_key,tr.citation_key))
    # This index includes only records used in fitting at least one model, not held-out cases.
    subset=retained[[ (a,b) in used for a,b in zip(retained.case_key,retained.citation_key)]].copy()
    def description(r):
        return ' '.join(str(r.get(c) or '') for c in ['Type of Matrimonial Proceeding','Ground of Divorce','Duration of Marriage',
               'Duration of Separation','Number of Children','Employment Status of Husband','Employment Status of Wife',
               'Independent Income of Husband','Independent Income of Wife'])
    entries=[]
    for _,r in subset.iterrows():
        entries.append({'case_name':r['Case Name'],'citation':r['Case Citation'],
                        'description':description(r),'case_key':r.case_key,'citation_key':r.citation_key})
    (DATA/'training_case_index.json').write_text(json.dumps(entries,ensure_ascii=False,default=str),encoding='utf-8')
    report['training_case_index_size']=len(entries)
    (DATA/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:{m:v for m,v in info.items() if m not in ('roc','confusion','per_class')} for k,info in report['models'].items()},indent=2))
    print('Training-only case index:',len(entries))

if __name__=='__main__':main()
