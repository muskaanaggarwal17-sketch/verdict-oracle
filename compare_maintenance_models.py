from pathlib import Path
import json, re, unicodedata, warnings
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge,HuberRegressor,QuantileRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,median_absolute_error,r2_score
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(__file__).resolve().parent
CATS=['Type of Matrimonial Proceeding','Ground of Divorce','Employment Status of Husband','Employment Status of Wife']
NUMS=['Marriage Duration Years','Separation Duration Years','Children Count','Husband Income INR','Wife Income INR']
FEATURES=CATS+NUMS

def read(n):return pd.DataFrame(json.loads((ROOT/'data'/n).read_text()))
def key(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s or '')).casefold()).strip()
def prepare(d):
 d=d.copy()
 for c in CATS:d[c]=d[c].fillna('Unknown').astype(str).replace({'':'Unknown'})
 for c in NUMS:
  if c not in d:d[c]=np.nan
  d[c]=pd.to_numeric(d[c],errors='coerce')
 d['case_key']=d['Case Name'].map(key);d['citation_key']=d['Case Citation'].map(key)
 return d[d.citation_key.ne('')].drop_duplicates(['case_key','citation_key']).copy()
def pipe(method,alpha=1):
 prep=ColumnTransformer([('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='Unknown')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CATS),('num',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value=-1,keep_empty_features=True)),('scale',StandardScaler())]),NUMS)],sparse_threshold=0)
 est={'ridge':Ridge(alpha=alpha),'huber':HuberRegressor(alpha=alpha,epsilon=1.35,max_iter=1000),'quantile':QuantileRegressor(quantile=.5,alpha=alpha,solver='highs')}[method]
 return Pipeline([('prep',prep),('model',est)])
ret=prepare(read('retained.json'));maint=prepare(read('maintenance.json'))
groups=ret.case_key.where(ret.case_key.ne(''),ret.citation_key)
ti,vi=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(ret,groups=groups))
train_groups=set(groups.iloc[ti]);test_groups=set(groups.iloc[vi]);assert not train_groups&test_groups
cpi={int(k):v for k,v in json.loads((ROOT/'data'/'india_cpi_annual.json').read_text())['values'].items()} if (ROOT/'data'/'india_cpi_annual.json').exists() else None
if cpi is None: raise FileNotFoundError('Missing data/india_cpi_annual.json')
cpi[2026]=cpi[2025]*cpi[2025]/cpi[2024]
allres={}
for period in ['monthly','lump_sum_or_settlement']:
 d=maint[maint['Award Periodicity'].eq(period)].copy()
 d['Amount Awarded INR']=pd.to_numeric(d['Amount Awarded INR'],errors='coerce')
 d['Judgment Year']=pd.to_numeric(d['Judgment Year'],errors='coerce')
 d['base']=d['Judgment Year'].map(cpi)
 d=d[d['Amount Awarded INR'].ge(0)&d['base'].gt(0)].copy()
 d['target']=d['Amount Awarded INR']*cpi[2026]/d['base']
 d['group']=d.case_key.where(d.case_key.ne(''),d.citation_key)
 tr=d[d.group.isin(train_groups)].copy();te=d[d.group.isin(test_groups)].copy()
 assert not set(tr.group)&set(te.group)
 print('\n',period,'train',len(tr),'test',len(te),'median',tr.target.median(),flush=True)
 candidates=[('ridge_log_a10','ridge',10,'log'),('ridge_log_a1','ridge',1,'log'),('ridge_log_a100','ridge',100,'log'),('ridge_raw_a10','ridge',10,'raw'),('huber_log_a0.1','huber',.1,'log'),('huber_log_a1','huber',1,'log'),('quantile_log_a0.01','quantile',.01,'log'),('quantile_log_a0.1','quantile',.1,'log')]
 folds=list(GroupKFold(n_splits=4).split(tr,groups=tr.group))
 scores=[]
 for name,method,alpha,scale in candidates:
  vals=[];med=[];errors=[]
  for a,b in folds:
   t=tr.iloc[a];v=tr.iloc[b]
   model=pipe(method,alpha)
   try:
    model.fit(t[FEATURES],np.log1p(t.target) if scale=='log' else t.target)
    pred=model.predict(v[FEATURES]);pred=np.maximum(0,np.expm1(np.clip(pred,-10,20))) if scale=='log' else np.maximum(0,pred)
    vals.append(mean_absolute_error(v.target,pred));med.append(median_absolute_error(v.target,pred))
   except Exception as e:errors.append(str(e)[:120]);break
  if len(vals)==4:scores.append({'name':name,'method':method,'alpha':alpha,'scale':scale,'cv_mae_mean':float(np.mean(vals)),'cv_mae_sd':float(np.std(vals,ddof=1)),'cv_medae_mean':float(np.mean(med)),'fold_mae':[float(x) for x in vals]})
  else:print('FAILED',name,errors,flush=True)
 scores.sort(key=lambda x:x['cv_mae_mean'])
 baseline=[]
 for a,b in folds:baseline.append(mean_absolute_error(tr.iloc[b].target,np.repeat(tr.iloc[a].target.median(),len(b))))
 best=scores[0]; original=next(x for x in scores if x['name']=='ridge_log_a10')
 # Choose only if CV improvement >= 10%, and compare with original on same untouched holdout (for transparency, no selection based on holdout)
 improve=(original['cv_mae_mean']-best['cv_mae_mean'])/original['cv_mae_mean']
 chosen=best if improve>=.10 else original
 model=pipe(chosen['method'],chosen['alpha']);model.fit(tr[FEATURES],np.log1p(tr.target) if chosen['scale']=='log' else tr.target)
 pred=model.predict(te[FEATURES]);pred=np.maximum(0,np.expm1(np.clip(pred,-10,20))) if chosen['scale']=='log' else np.maximum(0,pred)
 original_model=pipe('ridge',10);original_model.fit(tr[FEATURES],np.log1p(tr.target))
 opred=np.maximum(0,np.expm1(np.clip(original_model.predict(te[FEATURES]),-10,20)))
 report={'period':period,'train_n':len(tr),'test_n':len(te),'cpi_2026_provisional':cpi[2026], 'training_median':float(tr.target.median()),'cv_median_baseline_mae':float(np.mean(baseline)), 'candidates':scores,'chosen':chosen['name'],'cv_improvement_vs_existing':float(improve),'holdout_selected_mae':float(mean_absolute_error(te.target,pred)),'holdout_existing_mae':float(mean_absolute_error(te.target,opred)),'holdout_selected_r2':float(r2_score(te.target,pred)),'holdout_existing_r2':float(r2_score(te.target,opred)),'holdout_selected_medae':float(median_absolute_error(te.target,pred))}
 allres[period]=report
 print('CV results',[(x['name'],round(x['cv_mae_mean']),round(x['cv_mae_sd'])) for x in scores],flush=True)
 print('SELECT',chosen['name'],'improve',round(improve,3),'holdout',round(report['holdout_selected_mae']), 'original',round(report['holdout_existing_mae']),flush=True)
 # Experimental model is not deployed or saved: avoid accidental model replacement.
(ROOT/'data'/'model_comparison.json').write_text(json.dumps(allres,indent=2))
