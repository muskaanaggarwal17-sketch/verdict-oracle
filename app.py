"""Verdict Oracle — deploy with Streamlit Community Cloud. No runtime Drive/Excel access."""
from pathlib import Path
import json,html,re,requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data';MODELS=ROOT/'models'
CATS=['Type of Matrimonial Proceeding','Ground of Divorce','Employment Status of Husband','Employment Status of Wife']
NUMS=['Marriage Duration Years','Separation Duration Years','Children Count','Husband Income INR','Wife Income INR']
FEATURES=CATS+NUMS
PERIOD_NAMES={'monthly':'Monthly','lump_sum_or_settlement':'Lump sum / settlement'}
st.set_page_config(page_title='Verdict Oracle | Matrimonial Case Research',page_icon='⚖️',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:linear-gradient(125deg,#f5f7ff 0%,#fbfaf6 58%,#edf7f5 100%);color:#30283c}
.block-container{padding-top:2rem;max-width:1190px}
h1,h2,h3{color:#163f54!important;letter-spacing:-.025em}
.hero{background:linear-gradient(115deg,#174d63,#387f80);padding:32px 36px;border-radius:22px;color:white;margin-bottom:20px;box-shadow:0 14px 32px #4b356425}
.hero h1{color:white!important;font-size:2.65rem;margin:0}.hero p{color:#f5e9f5;font-size:1.07rem;margin:.7rem 0 0}
.tag{display:inline-block;background:#e9dfef;border-radius:100px;color:#5b3d65;padding:6px 12px;font-weight:600;font-size:.82rem}
.note{background:#fff;border:1px solid #e8ddec;border-left:5px solid #4fa8a2;padding:15px 18px;border-radius:12px;margin:12px 0}
[data-testid="stMetric"]{background:white;border:1px solid #e9dfea;border-radius:14px;padding:16px}
div.stButton>button[kind="primary"]{background:#156b73;color:white;border:0;border-radius:12px;padding:12px 22px}
[data-testid="stTabs"] button{font-weight:650}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=86400,show_spinner=False)
def india_cpi():
    """Annual all-India CPI, World Bank FP.CPI.TOTL; no invented interpolation."""
    path=DATA/'india_cpi_annual.json'
    if path.exists():
        try:
            payload=json.loads(path.read_text(encoding='utf-8'))
            if payload.get('values'):return {int(y):float(v) for y,v in payload['values'].items()},payload.get('source','World Bank FP.CPI.TOTL')
        except (ValueError,TypeError,KeyError):pass
    try:
        url='https://api.worldbank.org/v2/country/IND/indicator/FP.CPI.TOTL?format=json&per_page=100'
        resp=requests.get(url,timeout=9);resp.raise_for_status()
        body=resp.json()
        values={int(x['date']):float(x['value']) for x in body[1] if x.get('value') is not None}
        if values:return values,'World Bank FP.CPI.TOTL (live API)'
    except (requests.RequestException,ValueError,IndexError,KeyError,TypeError):pass
    return {},'CPI data unavailable; run update_india_cpi.py once and commit its output'

def adjust_inr(amount,from_year,to_year,values):
    a=values.get(int(from_year));b=values.get(int(to_year))
    return amount*b/a if a and b and a>0 else None

@st.cache_data
def data():
    return (pd.DataFrame(json.loads((DATA/'divorce.json').read_text())),
            pd.DataFrame(json.loads((DATA/'maintenance.json').read_text())),
            json.loads((DATA/'evaluation.json').read_text()),
            json.loads((DATA/'training_case_index.json').read_text()))
@st.cache_resource
def load_models():
    return {p.stem:joblib.load(p) for p in MODELS.glob('*.joblib')}
@st.cache_resource
def retrieval_index():
    entries=json.loads((DATA/'training_case_index.json').read_text())
    # Optional source-verified PDF excerpts added by build_pdf_index.py, only for training cases.
    pdf_path=DATA/'verified_pdf_excerpts.json'
    if pdf_path.exists():
        pdfs=json.loads(pdf_path.read_text(encoding='utf-8'))
        lookup={(p['case_key'],p['citation_key']):p for p in pdfs}
        for e in entries:
            p=lookup.get((e['case_key'],e['citation_key']))
            if p:
                e['pdf_excerpt']=p['excerpt'];e['pdf_verified']=True;e['pdf_url']=p.get('pdf_url')
    docs=[e['description']+' '+e.get('pdf_excerpt','')[:2500] for e in entries]
    if not docs:return entries,None,None
    v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=35000)
    return entries,v,v.fit_transform(docs)

div,maint,report,case_index=data();models=load_models()
st.markdown('''<div class="hero"><div style="font-size:.85rem;letter-spacing:.13em;font-weight:700;color:#d5eeee">LEGAL DATA SCIENCE · RESEARCH PROTOTYPE</div><h1>⚖️ Verdict Oracle</h1><p>Exploring divorce outcomes, maintenance awards and comparable matrimonial judgments from the Punjab & Haryana High Court.</p></div>''',unsafe_allow_html=True)
st.caption('Built from the preprocessed Punjab & Haryana High Court corpus. Research only — not legal advice or an individual court-outcome guarantee.')

@st.dialog('Describe your matrimonial dispute',width='large')
def case_dialog():
    st.caption('Enter whatever is known. Every field is optional; the model can handle missing structured facts.')
    with st.form('facts_form'):
        c1,c2=st.columns(2,gap='large')
        with c1:
            proceeding=st.text_input('Type of proceeding',placeholder='e.g., divorce petition')
        with c2:
            ground=st.text_input('Ground for divorce',placeholder='e.g., cruelty, desertion')
        with c1:
            marriage=st.number_input('Marriage duration (years)',min_value=0.,max_value=100.,value=None,step=.5)
        with c2:
            separation=st.number_input('Separation duration (years)',min_value=0.,max_value=100.,value=None,step=.5)
        with c1:
            children=st.number_input('Number of children',min_value=0,max_value=30,value=None,step=1)
        with c2:
            period=st.selectbox('Maintenance award period to estimate',options=list(PERIOD_NAMES),format_func=lambda x:PERIOD_NAMES[x])
        with c1:
            hwork=st.text_input('Husband employment status',placeholder='e.g., salaried')
        with c2:
            wwork=st.text_input('Wife employment status',placeholder='e.g., unemployed')
        with c1:
            hincome=st.number_input('Husband income (INR; dataset period may vary)',min_value=0.,value=None,step=1000.)
        with c2:
            wincome=st.number_input('Wife income (INR; dataset period may vary)',min_value=0.,value=None,step=1000.)
        with c1:
            litigation=st.text_area('Previous litigation outcome (optional)',placeholder='e.g., Family Court dismissed divorce petition',height=95)
        with c2:
            sought=st.number_input('Maintenance/alimony sought (₹; optional)',min_value=0.,value=None,step=1000.)
            sought_period=st.selectbox('Maintenance sought payment basis',['Monthly','Lump sum / settlement','Not specified'])
        narrative=st.text_area('Background of the dispute',height=150,placeholder='Describe the chronology, conduct alleged, children, finances and relief sought. Do not include a final judgment.')
        submitted=st.form_submit_button('Analyse this case',type='primary',use_container_width=True)
    if submitted:
        f=dict(zip(FEATURES,[proceeding,ground,hwork,wwork,marriage,separation,children,hincome,wincome]))
        known=sum(v is not None and str(v).strip()!='' for v in f.values())
        if known==0 and not narrative.strip():st.error('Please enter at least one case detail.');return
        st.session_state.case={'features':f,'narrative':narrative,'period':period,'known':known,'litigation':litigation,'sought':sought,'sought_period':sought_period}
        st.rerun()

c1,c2=st.columns([2.6,1])
with c1:st.markdown('### Case assessment')
with c2:
    if st.button('＋ New case assessment',type='primary',use_container_width=True):case_dialog()
if 'case' not in st.session_state:
    st.markdown('<div class="note">Select <b>New case assessment</b> to open the case-details dialogue.</div>',unsafe_allow_html=True)

T=st.tabs(['⚖️ Final prediction','🧭 How the model reached it','📚 Comparable case laws','📈 Model accuracy','📊 Historical trends'])
case=st.session_state.get('case')

def features_df(f):
    return pd.DataFrame([{k:(v if v is not None and str(v).strip()!='' else ('Unknown' if k in CATS else np.nan)) for k,v in f.items()}])[FEATURES]

def similar(f,narrative,k=5):
    entries,v,matrix=retrieval_index()
    if v is None:return []
    query=' '.join(str(f.get(c) or '') for c in CATS)+' '+str(narrative or '')
    if not query.strip():return []
    scores=cosine_similarity(v.transform([query]),matrix).ravel()
    return [(entries[i],float(scores[i])) for i in np.argsort(-scores) if scores[i]>0 and entries[i].get('pdf_verified')][:k]

with T[0]:
    st.subheader('Final prediction')
    if case is None:st.info('Enter case details to see a prediction.')
    elif case['known']==0:
        st.warning('Only narrative text was supplied. The current supervised models require at least one structured fact; the narrative can still retrieve similar judgments. Please add a structured field to obtain numerical predictions.')
    else:
        f=case['features'];X=features_df(f);period=case['period']
        if case['known']<5:st.warning(f'Only {case["known"]} of 9 structured fields were provided. Missing values are imputed; predictions may be less informative. More accurate and complete facts may help, but accuracy is not guaranteed.')
        a,b=st.columns(2)
        with a:
            if 'divorce' in models:
                m=models['divorce'];p=m.predict_proba(X)[0];classes=m.classes_
                predicted=str(m.predict(X)[0]);st.metric('Predicted divorce outcome',predicted.title())
                st.caption('Model scores: '+', '.join(f'{cl.title()} {100*float(pr):.1f}%' for cl,pr in zip(classes,p)))
                st.caption('Scores are not calibrated probabilities of what a court will decide.')
            else:st.error('Divorce model unavailable.')
        with b:
            m=models.get('maintenance_'+period)
            if m is not None:
                amount=float(max(0,np.expm1(np.clip(m.predict(X)[0],-10,20))))
                st.metric('Estimated maintenance / alimony',f'₹{amount:,.0f}')
                st.caption('Payment basis: '+PERIOD_NAMES[period]+'. '+(f"CPI basis: {report['maintenance_cpi']['target_year']} ({report['maintenance_cpi'].get('status','reference index')}). " if report.get('maintenance_cpi',{}).get('enabled') else 'Historical nominal basis (retraining pending). ')+'Point estimate only; no validated prediction interval.')
            else:st.info('Insufficient labelled data for the selected payment period. No reliable estimate available.')
        st.markdown('<div class="note"><b>Interpretation:</b> These are statistical estimates from selected historical judgments. Comparable judgments do not mathematically determine the prediction. Maintenance training amounts are adjusted to a common CPI year only when the CPI-retrained model is deployed; judgment year is not an input feature.</div>',unsafe_allow_html=True)
        if case.get('sought') is not None:
            st.caption(f"Maintenance sought by applicant: ₹{case['sought']:,.0f} ({case.get('sought_period','unspecified')}). This field is displayed for context and is NOT a feature in the currently trained models.")
        if case.get('litigation'):
            st.caption('Previous litigation outcome (context only): '+case['litigation'])
        cpi_meta=report.get('maintenance_cpi',{})
        if cpi_meta.get('enabled'):
            st.caption(f"Maintenance estimates use awards normalised to {cpi_meta['target_year']} CPI purchasing power ({cpi_meta.get('status','reference index')}) during model training. These are statistical estimates, not current court-award guarantees.")
        else:
            st.warning('Maintenance models have not yet been retrained with CPI-normalised awards. Run the included one-time retraining notebook and deploy the updated model and evaluation files.')
with T[1]:
    st.subheader('How the prediction was produced')
    st.write('**Divorce outcome:** Logistic Regression compares the details you provide with patterns in previously decided cases and estimates whether divorce is more likely to be granted or refused.')
    st.write('**Maintenance and alimony:** Separate Ridge Regression models estimate monthly awards or lump-sum settlements. Historical awards used for training have been converted to a common provisional 2026 purchasing-power basis.')
    st.write('**Incomplete details:** You can leave fields blank. The model handles missing information, but results based on fewer facts require greater caution.')
    st.write('**Comparable judgments:** Matching PDF-verified judgments are shown separately as research references. Their similarity scores do not determine the statistical prediction.')
    if case:
        st.dataframe(pd.DataFrame([{'Feature':k,'Provided value':str(v) if v is not None and str(v).strip() else 'Missing — imputed'} for k,v in case['features'].items()]),hide_index=True,use_container_width=True)
        if case['known'] and 'divorce' in models:
            try:
                m=models['divorce'];z=m.named_steps['prep'].transform(features_df(case['features']))
                if hasattr(z,'toarray'):z=z.toarray()
                names=m.named_steps['prep'].get_feature_names_out()
                values=(z[0]*m.named_steps['model'].coef_[0]);pairs=sorted(zip(names,values),key=lambda x:abs(x[1]),reverse=True)[:8]
                st.write(f'**Largest model contributions toward class “{m.classes_[1]}”** (log-odds scale):')
                st.dataframe(pd.DataFrame(pairs,columns=['Encoded feature','Contribution']),hide_index=True,use_container_width=True)
                st.caption('These are mathematical feature contributions, not legal reasoning or evidence of causation.')
            except Exception as exc:st.caption('Feature breakdown unavailable: '+str(exc))
    if case:
        st.write('**Additional context, not included in the existing trained predictors:**')
        st.write('Previous litigation: '+(case.get('litigation') or 'Not supplied'))
        st.write('Maintenance sought: '+(f"₹{case['sought']:,.0f} ({case['sought_period']})" if case.get('sought') is not None else 'Not supplied'))
with T[2]:
    st.subheader('Comparable judgments — separate research evidence')
    st.caption('Search restricted to case records in the training partition and actually used to fit a model. Similarity is lexical overlap, not precedential authority.')
    if not case:st.info('Enter case facts to search comparable cases.')
    else:
        search_text=' '.join([case['narrative'],case.get('litigation') or '',f"maintenance sought {case.get('sought')} {case.get('sought_period')}" if case.get('sought') is not None else ''])
        results=similar(case['features'],search_text)
        if not results:st.info('No verified full-text judgments match the supplied description in the available evidence index.')
        for i,(r,score) in enumerate(results,1):
            with st.expander(f'{i}. {r["case_name"]} · {r["citation"]} · similarity {score:.2f}',expanded=i==1):
                if r.get('pdf_verified'):
                    st.success('Excerpt verified against a matching judgment PDF in the optional PDF index.')
                    st.write(r['pdf_excerpt'][:1800])
                    if r.get('pdf_url'): st.link_button('Open verified judgment PDF in Google Drive',r['pdf_url'])
with T[3]:
    st.subheader('Held-out result analysis')
    st.caption('The main results use judgments held out from training (127 divorce cases). Additional cross-validation results below evaluate the training partition across five non-overlapping case groups. Neither metric guarantees accuracy for a new case.')
    d=report['models'].get('divorce',{})
    if d.get('status')=='trained':
        a,b,c,dcol=st.columns(4)
        a.metric('Accuracy',f'{d["accuracy"]:.1%}');b.metric('Macro F1',f'{d["macro_f1"]:.3f}')
        c.metric('ROC–AUC',f'{d["roc_auc"]:.3f}' if d['roc_auc'] is not None else 'N/A');dcol.metric('Test judgments',str(d['test_n']))
        left,right=st.columns(2)
        with left:
            roc=d['roc'];fig=go.Figure()
            fig.add_trace(go.Scatter(x=roc['fpr'],y=roc['tpr'],mode='lines',name='Classifier',line=dict(color='#8056a0',width=3)))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Chance',line=dict(dash='dash',color='#a7a0ac')))
            fig.update_layout(title='ROC curve — granted as positive class',xaxis_title='False positive rate',yaxis_title='True positive rate',height=350,margin=dict(t=55,b=30))
            st.plotly_chart(fig,use_container_width=True)
        with right:
            cm=d['confusion'];fig=px.imshow(cm,text_auto=True,x=['Predicted granted','Predicted refused'],y=['Actual granted','Actual refused'],color_continuous_scale='Purples',title='Held-out confusion matrix')
            fig.update_layout(height=350,margin=dict(t=55,b=30));st.plotly_chart(fig,use_container_width=True)
        st.dataframe(pd.DataFrame(d['per_class']).T,use_container_width=True)
        cv=report.get('divorce_training_cross_validation')
        if cv:
            st.markdown('#### Five-fold cross-validation (training judgments only)')
            st.caption('Five separate evaluations within the training partition. The original 127 held-out judgments were not used in these folds.')
            cvrows=[{'Metric':name,'Mean':round(values['mean'],3),'Standard deviation':round(values['sd'],3)} for name,values in [('Accuracy',cv['accuracy']),('Macro F1',cv['macro_f1']),('ROC–AUC',cv['roc_auc'])]]
            st.dataframe(pd.DataFrame(cvrows),hide_index=True,use_container_width=True)
    st.markdown('#### Maintenance prediction error')
    rows=[]
    for period in PERIOD_NAMES:
        v=report['models'].get('maintenance_'+period,{})
        rows.append({'Payment period':PERIOD_NAMES[period],'Status':v.get('status'),'Train n':v.get('train_n',0),'Test n':v.get('test_n',0),'MAE (INR)':v.get('mae_inr'),'R²':v.get('r2')})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.warning('Maintenance estimates are based on small labelled samples (8 monthly and 13 lump-sum test cases). Negative or near-zero R² indicates limited predictive reliability. ROC and F1 apply to divorce classification; MAE and R² apply to maintenance regression.')
    # Optional reproducible experiment; the deployed predictors are unchanged.
    comparison_path=DATA/'model_comparison.json'
    if comparison_path.exists():
        comparison=json.loads(comparison_path.read_text(encoding='utf-8'))
        st.markdown('#### Experimental maintenance model comparison')
        st.caption('Four-fold case-grouped cross-validation on the original training partition. The holdout was kept separate. These experimental models are NOT deployed; figures are exploratory because the samples are small and some folds contain extreme awards.')
        for period,label in PERIOD_NAMES.items():
            entry=comparison.get(period,{})
            if not entry:continue
            st.markdown('**'+label+'**')
            rows=[{'Candidate':v['name'].replace('_',' '),'CV MAE (₹)':round(v['cv_mae_mean']),'Fold-to-fold SD (₹)':round(v['cv_mae_sd'])} for v in entry.get('candidates',[])]
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            # Fold-level diagnostics reveal when one unstable fold dominates the mean.
            with st.expander('Inspect individual validation folds and instability'):
                fold_rows=[]
                for candidate in entry.get('candidates',[]):
                    errors=candidate.get('fold_mae',[])
                    if not errors:continue
                    fold_rows.append({'Candidate':candidate['name'].replace('_',' '),
                        **{f'Fold {i+1} MAE (₹)':round(float(value)) for i,value in enumerate(errors)},
                        'Worst/best fold ratio':round(max(errors)/max(min(errors),1),1)})
                if fold_rows:st.dataframe(pd.DataFrame(fold_rows),hide_index=True,use_container_width=True)
                st.caption('A large worst/best fold ratio indicates unstable validation performance. The comparison records errors, not the individual predicted amounts; those require rerunning the audit script with the repository datasets.')
            audit_path=DATA/'model_comparison_stability_audit.json'
            if audit_path.exists():
                try:
                    audit=json.loads(audit_path.read_text(encoding='utf-8')).get(period,{})
                    if audit:
                        with st.expander('Verified stability audit: extreme prediction diagnostics'):
                            audit_rows=[]
                            for candidate in audit.get('candidates',[]):
                                folds=candidate.get('fold_diagnostics',[])
                                audit_rows.append({
                                    'Candidate':candidate['name'].replace('_',' '),
                                    'Largest predicted amount (₹)':round(max((f.get('predicted_max',0) for f in folds),default=0)),
                                    'Predictions above fold training maximum':sum(f.get('count_predicted_above_training_max',0) for f in folds),
                                    'Log predictions clipped':sum(f.get('count_log_clipped',0) for f in folds),
                                })
                            if audit_rows:st.dataframe(pd.DataFrame(audit_rows),hide_index=True,use_container_width=True)
                            st.caption('Audit reproduced from the supplied original data with provisional 2026 CPI. Very large values arise in particular validation folds, especially after exponentiating log-scale predictions. These experimental models are not deployed.')
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    st.caption(f'Additional stability audit unavailable: {exc}')
            st.caption(f"Median baseline CV MAE: ₹{entry['cv_median_baseline_mae']:,.0f}. Existing Ridge holdout MAE: ₹{entry['holdout_existing_mae']:,.0f}. Experimental candidate holdout MAE: ₹{entry['holdout_selected_mae']:,.0f}. The experiment did not justify replacing the deployed model.")
        st.info('A lower cross-validation error does not guarantee better prediction on unseen judgments. Extremely large errors in log-target models may arise when predictions are exponentiated; the audit script checks this explicitly. Existing maintenance models remain unchanged; no award amounts have been artificially increased.')
    diagnostic=report.get('maintenance_diagnostics',{})
    if diagnostic:
        st.markdown('#### Comparison with a simple median-award baseline')
        baseline_rows=[]
        for p,label in PERIOD_NAMES.items():
            x=diagnostic.get(p)
            if x:baseline_rows.append({'Payment period':label,'Ridge MAE (₹)':round(x['model_mae']),'Median baseline MAE (₹)':round(x['median_baseline_mae']),'Training median (2026 ₹)':round(x['train_median_2026'])})
        st.dataframe(pd.DataFrame(baseline_rows),hide_index=True,use_container_width=True)
        st.caption('The baseline always predicts the median award in the training data. A lower MAE is better. This is a diagnostic, not a claim of reliable individual-case prediction.')
with T[4]:
    st.subheader('Historical patterns in the supplied dataset')
    st.caption('These plots describe the selected labelled judgments, not all High Court cases. Missing labels are excluded. The CPI-adjusted view is descriptive, not a model forecast.')
    a,b=st.columns(2)
    with a:
        counts=div['Divorce Label'].value_counts().rename_axis('Outcome').reset_index(name='Judgments')
        st.plotly_chart(px.bar(counts,x='Outcome',y='Judgments',color='Outcome',color_discrete_map={'granted':'#7b5792','refused':'#d8a95b'},title='Most frequent recorded divorce outcomes'),use_container_width=True)
    with b:
        d=div.copy();d['Judgment Year']=pd.to_numeric(d.get('Judgment Year'),errors='coerce')
        d=d[d['Judgment Year'].between(1990,2026)]
        counts=d.groupby(['Judgment Year','Divorce Label']).size().reset_index(name='Judgments')
        st.plotly_chart(px.line(counts,x='Judgment Year',y='Judgments',color='Divorce Label',markers=True,title='Recorded outcomes by year',color_discrete_map={'granted':'#7b5792','refused':'#d8a95b'}),use_container_width=True)
    d=maint.copy();d['Judgment Year']=pd.to_numeric(d.get('Judgment Year'),errors='coerce');d['Amount Awarded INR']=pd.to_numeric(d.get('Amount Awarded INR'),errors='coerce')
    d=d[d['Judgment Year'].between(1990,2026)&d['Amount Awarded INR'].ge(0)]
    if len(d):
        st.markdown('#### Recorded maintenance and alimony awards')
        st.caption('Monthly and lump-sum payments are shown separately because their amounts are not directly comparable.')
        v=d.groupby(['Judgment Year','Award Periodicity'])['Amount Awarded INR'].median().reset_index()
        cpi,cpi_source=india_cpi()
        cpi_meta=report.get('maintenance_cpi',{})
        target_year=cpi_meta.get('target_year')
        if target_year==2026 and 2026 not in cpi and 2024 in cpi and 2025 in cpi and cpi[2024]>0:
            cpi[2026]=cpi[2025]*(cpi[2025]/cpi[2024])
            cpi_source+='; provisional 2026 extrapolation (not observed)'
        if not cpi:
            st.warning('CPI reference data are unavailable. Showing historical nominal amounts only.')
        else:
            # Match the CPI target year used to train the deployed Ridge models.
            if cpi_meta.get('target_year') != 2026:
                st.warning('The evaluation metadata do not specify the expected 2026 CPI reference; the comparison chart is unavailable.')
            else:
                if 2026 not in cpi:
                    if 2024 in cpi and 2025 in cpi and cpi[2024]>0:
                        cpi[2026]=cpi[2025]*(cpi[2025]/cpi[2024])
                    else:
                        st.warning('2024/2025 CPI observations needed for the provisional 2026 comparison are missing.')
                if 2026 in cpi:
                    d['2026-adjusted INR']=d.apply(lambda r:adjust_inr(r['Amount Awarded INR'],int(r['Judgment Year']),2026,cpi),axis=1)
        periods=[p for p in PERIOD_NAMES if p in set(d['Award Periodicity'].dropna())]
        if periods:
            choice=st.selectbox('Award payment type to visualise',periods,format_func=lambda x:PERIOD_NAMES[x])
            subset=d[d['Award Periodicity'].eq(choice)].copy()
            summary=subset.groupby('Judgment Year',as_index=False).agg(
                nominal=('Amount Awarded INR','median'),
                cases=('Amount Awarded INR','size'))
            if '2026-adjusted INR' in subset:
                adj=subset.groupby('Judgment Year',as_index=False)['2026-adjusted INR'].median()
                summary=summary.merge(adj,on='Judgment Year',how='left')
            summary=summary.sort_values('Judgment Year')
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=summary['Judgment Year'],y=summary['nominal'],name='Original recorded award',mode='lines+markers',line=dict(color='#156b73',width=2.5),customdata=summary['cases'],hovertemplate='Year %{x}<br>Median ₹%{y:,.0f}<br>Cases %{customdata}<extra></extra>'))
            if '2026-adjusted INR' in summary:
                fig.add_trace(go.Scatter(x=summary['Judgment Year'],y=summary['2026-adjusted INR'],name='Provisional 2026 purchasing power',mode='lines+markers',line=dict(color='#8a5fa0',width=2.5),customdata=summary['cases'],hovertemplate='Year %{x}<br>Median ₹%{y:,.0f}<br>Cases %{customdata}<extra></extra>'))
            fig.update_layout(title=f'Median {PERIOD_NAMES[choice].lower()} award: original vs 2026-adjusted',height=430,xaxis_title='Judgment year',yaxis_title='Median awarded INR',legend=dict(orientation='h',y=1.12),margin=dict(t=90,b=35))
            st.plotly_chart(fig,use_container_width=True)
            st.caption('Each point is the median of the available awards for that year. Purple uses the same provisional 2026 CPI reference as the retrained maintenance models; it is not an observed 2026 annual CPI. Years with few awards may fluctuate sharply.')
            st.markdown('#### Distribution of recorded maintenance and alimony awards')
            st.caption('Recorded awards only, not model predictions. Decade is the default; individual years are shown as observations rather than unreliable small-sample boxplots.')
            c1,c2,c3=st.columns([1.3,1,1])
            with c1:
                amount_view=st.radio('Amounts shown', ['Provisional 2026 CPI-adjusted', 'Original nominal'], key='award_box_amount')
            with c2:
                scale_view=st.radio('Y-axis scale', ['Linear', 'Log (zero-safe)'], key='award_box_scale', help='Log view uses log(1 + amount), so genuine zero awards remain visible.')
            with c3:
                grouping=st.selectbox('Group awards by', ['Decade', 'All judgments', 'Judgment year'], key='award_box_group')
            amount_col='2026-adjusted INR' if amount_view.startswith('Provisional') and '2026-adjusted INR' in subset.columns else 'Amount Awarded INR'
            if amount_view.startswith('Provisional') and amount_col!='2026-adjusted INR':
                st.warning('2026-adjusted award data are unavailable; displaying nominal awards instead.')
            boxes=subset.dropna(subset=[amount_col]).copy()
            boxes=boxes[np.isfinite(pd.to_numeric(boxes[amount_col],errors='coerce')) & (boxes[amount_col]>=0)]
            if not boxes.empty:
                boxes['Award group']=('All recorded judgments' if grouping=='All judgments' else
                    boxes['Judgment Year'].floordiv(10).mul(10).astype(int).astype(str)+'s' if grouping=='Decade' else
                    boxes['Judgment Year'].astype(int).astype(str))
                groups=sorted(boxes['Award group'].unique(),key=lambda x:int(x[:4]) if x[:4].isdigit() else 0)
                counts_box=boxes.groupby('Award group')[amount_col].agg(['count','median']).reindex(groups).reset_index()
                counts_box=counts_box.rename(columns={'count':'Judgments','median':'Median award (₹)'})
                labels={g:f'{g} (n={int((boxes["Award group"]==g).sum())})' for g in groups}
                log_view=scale_view.startswith('Log')
                display=lambda values:np.log1p(np.asarray(values,dtype=float)) if log_view else np.asarray(values,dtype=float)
                fig_box=go.Figure()
                if grouping=='Judgment year':
                    # Do not imply meaningful quartiles for years containing only one or two cases.
                    fig_box.add_trace(go.Scatter(
                        x=boxes['Judgment Year'].astype(int),y=display(boxes[amount_col]),
                        mode='markers',name='Recorded awards',
                        marker=dict(color='#156b73',size=8,opacity=.72),
                        customdata=np.column_stack((boxes[amount_col].to_numpy(),boxes['Judgment Year'].astype(int).to_numpy())),
                        hovertemplate='Year %{customdata[1]}<br>Award: ₹%{customdata[0]:,.0f}<extra></extra>'))
                    yearly=boxes.groupby('Judgment Year')[amount_col].median().sort_index()
                    if len(yearly)>1:
                        fig_box.add_trace(go.Scatter(x=yearly.index.astype(int),y=display(yearly.values),
                            mode='lines+markers',name='Yearly median (descriptive)',
                            line=dict(color='#8a5fa0',width=2,dash='dot'),marker=dict(size=5),
                            customdata=yearly.values,
                            hovertemplate='Year %{x}<br>Yearly median: ₹%{customdata:,.0f}<extra></extra>'))
                    fig_box.update_xaxes(title='Judgment year',dtick=2)
                else:
                    for g in groups:
                        vals=boxes.loc[boxes['Award group']==g,amount_col].astype(float).to_numpy()
                        q1,median,q3=np.percentile(vals,[25,50,75])
                        fig_box.add_trace(go.Box(
                            x=[labels[g]]*len(vals),y=display(vals),name=labels[g],
                            boxpoints='all',jitter=.32,pointpos=-1.35,
                            marker=dict(color='#156b73',size=6,opacity=.65),
                            line=dict(color='#156b73'),fillcolor='rgba(21,107,115,.16)',
                            customdata=vals,
                            hovertemplate='Award: ₹%{customdata:,.0f}<extra></extra>',showlegend=False))
                        # Separate median marker provides a consistently formatted statistical tooltip.
                        fig_box.add_trace(go.Scatter(x=[labels[g]],y=display([median]),mode='markers',
                            marker=dict(symbol='diamond',size=9,color='#8a5fa0'),showlegend=False,
                            hovertemplate=(f'Median: ₹{median:,.0f}<br>Q1: ₹{q1:,.0f}<br>Q3: ₹{q3:,.0f}'
                                           f'<br>Minimum: ₹{vals.min():,.0f}<br>Maximum: ₹{vals.max():,.0f}'
                                           f'<br>Judgments: {len(vals)}<extra></extra>')))
                    fig_box.update_xaxes(title='Judgment period',categoryorder='array',categoryarray=[labels[g] for g in groups])
                if log_view:
                    max_value=float(boxes[amount_col].max())
                    candidate_ticks=[0,100,500,1000,2000,5000,10000,20000,50000,100000,250000,500000,1000000,5000000,10000000,50000000]
                    ticks=[v for v in candidate_ticks if v<=max_value*1.15]
                    if not ticks: ticks=[0]
                    fig_box.update_yaxes(tickmode='array',tickvals=np.log1p(ticks).tolist(),ticktext=[f'₹{v:,.0f}' for v in ticks],title='Award (₹, log1p scale)')
                else:
                    fig_box.update_yaxes(title='Award (₹)',tickprefix='₹',tickformat=',.0f')
                fig_box.update_layout(title=f'Distribution of {PERIOD_NAMES[choice].lower()} awards',
                    height=470,margin=dict(t=65,b=55),hovermode='closest',showlegend=(grouping=='Judgment year'))
                st.plotly_chart(fig_box,use_container_width=True)
                if grouping=='Judgment year':
                    st.caption(f'{len(boxes)} recorded awards shown as individual points. The dotted line connects yearly medians for orientation, not a fitted predictive trend. Years with one observation have a median equal to that observation.')
                else:
                    st.caption(f'{len(boxes)} recorded awards included. Each dot is an actual recorded award. Groups with fewer than five judgments have limited interpretive value. Box whiskers use 1.5 × IQR; points outside them are not necessarily data errors.')
                if log_view:
                    st.caption('Log display is for visualisation only (log1p), and does not transform the underlying awards or change any model. Axis labels and hover amounts remain in rupees.')
                with st.expander('Distribution sample sizes and medians'):
                    st.dataframe(counts_box,hide_index=True,use_container_width=True)
            with st.expander('View the values and sample sizes behind the chart'):
                st.dataframe(summary.rename(columns={'Judgment Year':'Year','nominal':'Median nominal award (₹)','2026-adjusted INR':'Median 2026-adjusted award (₹)','cases':'Judgments'}),hide_index=True,use_container_width=True)
    st.caption(f'Preprocessed source: {report["retained_rows"]:,} retained records; {report["divorce_labeled_rows"]:,} divorce labels; {report["maintenance_labeled_rows"]:,} monetary award records. The year-specific samples may be small.')

st.divider()
st.caption('Verdict Oracle · Research prototype · Dataset and models bundled for reproducible access · Not a substitute for professional legal assessment.')
