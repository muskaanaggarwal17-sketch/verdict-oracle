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
PERIOD_NAMES={'monthly':'Monthly','lump_sum_or_settlement':'Lump sum / settlement','annual':'Annual'}
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
        c1,c2=st.columns(2)
        with c1:
            proceeding=st.text_input('Type of proceeding',placeholder='e.g., divorce petition')
            ground=st.text_input('Ground for divorce',placeholder='e.g., cruelty, desertion')
            marriage=st.number_input('Marriage duration (years)',min_value=0.,max_value=100.,value=None,step=.5)
            separation=st.number_input('Separation duration (years)',min_value=0.,max_value=100.,value=None,step=.5)
            children=st.number_input('Number of children',min_value=0,max_value=30,value=None,step=1)
        with c2:
            hwork=st.text_input('Husband employment status',placeholder='e.g., salaried')
            wwork=st.text_input('Wife employment status',placeholder='e.g., unemployed')
            hincome=st.number_input('Husband income (INR; dataset period may vary)',min_value=0.,value=None,step=1000.)
            wincome=st.number_input('Wife income (INR; dataset period may vary)',min_value=0.,value=None,step=1000.)
            period=st.selectbox('Maintenance award period to estimate',options=list(PERIOD_NAMES),format_func=lambda x:PERIOD_NAMES[x])
            litigation=st.text_area('Previous litigation outcome (optional)',placeholder='e.g., Family Court dismissed divorce petition; appeal pending',height=85)
            sought=st.number_input('Maintenance/alimony sought (₹; optional)',min_value=0.,value=None,step=1000.)
            sought_period=st.selectbox('Maintenance sought payment basis', ['Monthly','Lump sum / settlement','Annual','Not specified'])
            comparison_year=st.number_input('Historical reference year for CPI illustration (optional)',min_value=1990,max_value=2026,value=None,step=1,help='This does not change the model prediction. Select only if you want to illustrate how the same rupee amount compares in purchasing power with the latest available CPI year.')
        narrative=st.text_area('Background of the dispute',height=150,placeholder='Describe the chronology, conduct alleged, children, finances and relief sought. Do not include a final judgment.')
        submitted=st.form_submit_button('Analyse this case',type='primary',use_container_width=True)
    if submitted:
        f=dict(zip(FEATURES,[proceeding,ground,hwork,wwork,marriage,separation,children,hincome,wincome]))
        known=sum(v is not None and str(v).strip()!='' for v in f.values())
        if known==0 and not narrative.strip():st.error('Please enter at least one case detail.');return
        st.session_state.case={'features':f,'narrative':narrative,'period':period,'known':known,'litigation':litigation,'sought':sought,'sought_period':sought_period,'comparison_year':comparison_year}
        st.rerun()

c1,c2=st.columns([2.6,1])
with c1:st.markdown('### Case assessment')
with c2:
    if st.button('＋ New case assessment',type='primary',use_container_width=True):case_dialog()
if 'case' not in st.session_state:
    st.markdown('<div class="note">Select <b>New case assessment</b> to open the case-details dialogue. The app is ready without Excel uploads or Google Drive authentication.</div>',unsafe_allow_html=True)

T=st.tabs(['⚖️ Final prediction','🧭 How the model reached it','📚 Comparable case laws','📈 Model accuracy','📊 Historical trends'])
case=st.session_state.get('case')

def features_df(f):
    return pd.DataFrame([{k:(v if v is not None and str(v).strip()!='' else ('Unknown' if k in CATS else np.nan)) for k,v in f.items()}])[FEATURES]

def similar(f,narrative,k=5,verified_only=False):
    entries,v,matrix=retrieval_index()
    if v is None:return []
    query=' '.join(str(f.get(c) or '') for c in CATS)+' '+str(narrative or '')
    if not query.strip():return []
    scores=cosine_similarity(v.transform([query]),matrix).ravel()
    return [(entries[i],float(scores[i])) for i in np.argsort(-scores) if scores[i]>0 and (not verified_only or entries[i].get('pdf_verified'))][:k]

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
                st.caption('Payment basis: '+PERIOD_NAMES[period]+'. Point estimate only; no validated prediction interval.')
            else:st.info('Insufficient labelled data for the selected payment period. No reliable estimate available.')
        st.markdown('<div class="note"><b>Interpretation:</b> These are statistical estimates from selected historical judgments. Comparable judgments do not mathematically determine the prediction. The monetary estimate is nominal, and no judgment year is a predictive feature.</div>',unsafe_allow_html=True)
        if case.get('sought') is not None:
            st.caption(f"Maintenance sought by applicant: ₹{case['sought']:,.0f} ({case.get('sought_period','unspecified')}). This field is displayed for context and is NOT a feature in the currently trained models.")
        if case.get('litigation'):
            st.caption('Previous litigation outcome (context only): '+case['litigation'])
        cpi,cpi_source=india_cpi()
        if cpi:
            target=max(cpi)
            st.markdown('#### Inflation / purchasing-power illustration')
            st.caption(f'Annual India CPI ({cpi_source}). Latest available complete year: {target}. No extrapolation to 2026.')
            if case.get('comparison_year') is not None and m is not None:
                ref=int(case['comparison_year']); equivalent=adjust_inr(amount,ref,target,cpi)
                if equivalent is not None:
                    st.metric(f'Illustrative {target} rupee equivalent of the nominal estimate (assuming {ref} reference)',f'₹{equivalent:,.0f}')
                    st.caption('This is a hypothetical CPI conversion of the estimated amount, NOT a retrained model output or a forecast of a court award. The model does not establish a reference year for its estimate.')
                else:st.warning('No CPI index is available for the selected reference year; no adjustment calculated.')
            else:st.caption('Select a historical reference year in the case form to see a clearly labelled CPI illustration. Without a defensible reference year, the predicted amount cannot be reliably inflation-adjusted.')
        else:st.warning('CPI source currently unavailable. Historical rupee equivalents are hidden rather than guessed.')
with T[1]:
    st.subheader('How the prediction was produced')
    st.write('The divorce classifier is **Logistic Regression**. Maintenance uses separate **Ridge Regression** models by payment period. These are not neural networks or an ensemble.')
    st.write('The models use nine pre-decision structured fields. Missing values are replaced using the preprocessing pipeline; case names, citations, awarded amounts and the final judgment are never fed in as predictive features.')
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
    st.info('The narrative, previous litigation and maintenance sought are used for case context/retrieval only, not as numerical prediction features. To make them influence predictions, the models must be retrained and re-evaluated on reliably coded pre-decision data.')
with T[2]:
    st.subheader('Comparable judgments — separate research evidence')
    st.caption('Search restricted to case records in the training partition and actually used to fit a model. Similarity is lexical overlap, not precedential authority.')
    if not case:st.info('Enter case facts to search comparable cases.')
    else:
        pdf_available=(DATA/'verified_pdf_excerpts.json').exists()
        verified_only=st.checkbox('Show only PDF-verified judgments',value=pdf_available,help='When selected, the app excludes summary-only cases. This may reduce the number of comparable judgments.')
        search_text=' '.join([case['narrative'],case.get('litigation') or '',f"maintenance sought {case.get('sought')} {case.get('sought_period')}" if case.get('sought') is not None else ''])
        results=similar(case['features'],search_text,verified_only=verified_only)
        if pdf_available:
            indexed=sum(bool(x.get('pdf_verified')) for x in retrieval_index()[0])
            st.caption(f'{indexed} of {len(retrieval_index()[0])} training case records have verified PDF excerpts. Unmatched cases are not assumed to have full-text evidence.')
        if not results:st.info('No sufficiently similar training cases found from the supplied description.')
        for i,(r,score) in enumerate(results,1):
            with st.expander(f'{i}. {r["case_name"]} · {r["citation"]} · similarity {score:.2f}',expanded=i==1):
                if r.get('pdf_verified'):
                    st.success('Excerpt verified against a matching judgment PDF in the optional PDF index.')
                    st.write(r['pdf_excerpt'][:1800])
                    if r.get('pdf_url'): st.link_button('Open verified judgment PDF in Google Drive',r['pdf_url'])
                else:
                    st.warning('Structured case-summary record only — the full judgment PDF has not been packaged or verified for this entry.')
                    st.write(r['description'][:1800])
    if not (DATA/'verified_pdf_excerpts.json').exists():
        st.caption('PDF text has not been imported into this copy. Run the one-time verified Drive cache importer, then redeploy. The professor never needs to authenticate.')
with T[3]:
    st.subheader('Held-out result analysis')
    st.caption('Scores below were computed once on a held-out 20% group split and saved with the model. They are dataset-level performance metrics, not accuracy for the current individual case.')
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
    st.markdown('#### Maintenance prediction error')
    rows=[]
    for period in PERIOD_NAMES:
        v=report['models'].get('maintenance_'+period,{})
        rows.append({'Payment period':PERIOD_NAMES[period],'Status':v.get('status'),'Train n':v.get('train_n',0),'Test n':v.get('test_n',0),'MAE (INR)':v.get('mae_inr'),'R²':v.get('r2')})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.warning('Maintenance evaluation is based on small test subsets, particularly for monthly awards. ROC and F1 apply to the binary divorce task; MAE and R² are appropriate for monetary regression. These scores do not establish legal validity or fairness.')
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
        mode=st.radio('Maintenance amounts', ['Nominal rupees','CPI-adjusted rupees (latest complete year)'],horizontal=True)
        value_col='Amount Awarded INR'
        if mode.startswith('CPI'):
            if cpi:
                target=max(cpi)
                d['CPI-adjusted INR']=d.apply(lambda r:adjust_inr(r['Amount Awarded INR'],int(r['Judgment Year']),target,cpi),axis=1)
                d=d[d['CPI-adjusted INR'].notna()]
                value_col='CPI-adjusted INR'
                st.caption(f'Adjusted to {target} purchasing-power equivalents using {cpi_source}; CPI values for missing years are not interpolated.')
            else:st.warning('CPI data unavailable; showing nominal awards only.')
        if len(d):
            v=d.groupby(['Judgment Year','Award Periodicity'])[value_col].median().reset_index()
            st.plotly_chart(px.line(v,x='Judgment Year',y=value_col,color='Award Periodicity',markers=True,facet_row='Award Periodicity',title='Median recorded award by year and payment period',height=520),use_container_width=True)
    st.caption(f'Preprocessed source: {report["retained_rows"]:,} retained records; {report["divorce_labeled_rows"]:,} divorce labels; {report["maintenance_labeled_rows"]:,} monetary award records. The year-specific samples may be small.')

st.divider()
st.caption('Verdict Oracle · Research prototype · Dataset and models bundled for reproducible access · Not a substitute for professional legal assessment.')
