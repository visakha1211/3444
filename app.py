# -------------- FULL APP  -----------------------------------------------
#  EcoWise Insight Studio - 2025 (with advanced visuals)
#  Place banner_ecowise.png next to this file.  Requirements above.
# -------------------------------------------------------------------------
import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; from base64 import b64encode

# === extra libs ===
import plotly.express as px, plotly.graph_objects as go, scikitplot as skplt
import shap, networkx as nx; from pyvis.network import Network
import io, tempfile, os, json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, r2_score,
                             mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ───────────────── page & theme ──────────────────────────────────────────
st.set_page_config(page_title="EcoWise Insight Studio", layout="wide")
# banner
bn = Path("banner_ecowise.png")
if bn.exists():
    st.markdown(f'<img src="data:image/png;base64,{b64encode(bn.read_bytes()).decode()}"'
                f'style="width:100%;height:auto;">', unsafe_allow_html=True)
# css
st.markdown("""
<style>
:root{--primary:#2ecc71;--accent:#1abc9c;--bg:#f7fdf9;--txt:#033e26;}
html,body{background:var(--bg)!important;color:var(--txt);}
h1,h2,h3,h4{color:var(--primary);}
button[kind="primary"]{background:var(--accent)!important;border-radius:8px;}
button[kind="primary"]:hover{background:#17a689!important;}
div[data-testid="metric-container"]{background:#ffffffdd;border:1px solid #e5f7ed;
border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
</style>
""",unsafe_allow_html=True)
st.title("🌿 EcoWise Market Feasibility Dashboard")

# ───── data load ─────
csv = st.sidebar.file_uploader("Upload CSV", type="csv")
@st.cache_data
def load(f): return pd.read_csv(f) if f else pd.read_csv("ecowise_full_arm_ready.csv")
df = load(csv)
st.sidebar.success(f"{df.shape[0]} rows  |  {df.shape[1]} columns")

# --- helpers ---
def num_df(x): return x.select_dtypes("number")
def dummies(x): return pd.get_dummies(x, drop_first=True)

# tiny cm & roc
def pretty_cm(cm,lbl):
    fig,ax=plt.subplots(figsize=(3,2));im=ax.imshow(cm,cmap="viridis")
    ax.set_xticks(range(len(lbl)));ax.set_yticks(range(len(lbl)))
    ax.set_xticklabels(lbl,fontsize=6);ax.set_yticklabels(lbl,fontsize=6)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center",color="white",fontsize=6)
    plt.tight_layout(pad=.2);st.pyplot(fig,use_container_width=False)

def plot_network(rules):
    nt=Network(height='450px',width='100%',bgcolor='#ffffff',directed=True)
    for _,row in rules.iterrows():
        src=row['antecedents'];dst=row['consequents']
        nt.add_node(src,src,title=src,color='#1abc9c')
        nt.add_node(dst,dst,title=dst,color='#27ae60')
        nt.add_edge(src,dst,value=row['lift'])
    tmp=Path(tempfile.mkdtemp())/"net.html";nt.show(str(tmp))
    st.components.v1.html(tmp.read_text(),height=480,scrolling=False)

# ——— plot theme for mpl
plt.rcParams.update({"axes.prop_cycle":plt.cycler(color=["#1abc9c","#16a085","#2ecc71","#27ae60"])})

# ───── Tabs layout ─────
tabs=st.tabs(["📊 Visuals","🤖 Classify","📍 Cluster","🔗 Rules","📈 Regression","✨ Advanced"])

# ===================== TAB 1 Visuals =====================================
with tabs[0]:
    st.header("Descriptive Insights")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Correlation Matrix")
        fig,ax=plt.subplots(figsize=(6,4))
        im=ax.imshow(num_df(df).corr(),cmap="viridis")
        ax.set_xticks(range(len(num_df(df).columns)));ax.set_xticklabels(num_df(df).columns,rotation=90,fontsize=6)
        ax.set_yticks(range(len(num_df(df).columns)));ax.set_yticklabels(num_df(df).columns,fontsize=6)
        fig.colorbar(im,fraction=.035);st.pyplot(fig,use_container_width=True)
    with c2:
        st.subheader("Income by Country")
        fig=px.histogram(df,x="household_income_usd",color="country",nbins=25,opacity=.6)
        fig.update_layout(height=350,xaxis_range=[105,df["household_income_usd"].quantile(.95)])
        st.plotly_chart(fig,use_container_width=True)

    kpi,box=c1,c2  # reuse columns
    with kpi:
        st.metric("Avg. Bill (USD)",f"{df['monthly_energy_bill_usd'].mean():.1f}")
        st.metric("Median WTP",f"{df['max_willingness_to_pay_usd'].median():.0f}")
        st.metric("Intent ≥ 'Maybe'",f"{(df['willing_to_purchase_12m']>0).mean()*100:.1f}%")
    with box:
        st.subheader("WTP vs Eco-Concern")
        fig=px.box(df,x="env_concern_score",y="max_willingness_to_pay_usd",height=350)
        st.plotly_chart(fig,use_container_width=True)

    # new visuals
    st.divider()
    st.subheader("Average WTP by Country (Choropleth)")
    iso={'India':'IND','UAE':'ARE','Singapore':'SGP'}
    map_df=df.groupby('country')['max_willingness_to_pay_usd'].mean().reset_index()
    map_df['iso']=map_df['country'].map(iso)
    fig=px.choropleth(map_df,locations='iso',color='max_willingness_to_pay_usd',
                      color_continuous_scale='greens',hover_name='country',
                      labels={'max_willingness_to_pay_usd':'Avg WTP USD'})
    fig.update_layout(height=300,margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Barrier Frequency by Country")
    barrier_cols=[c for c in df.columns if c.startswith('barrier_')]
    bar_df=(df.groupby('country')[barrier_cols].sum()
              .rename(columns=lambda x:x.replace('barrier_',''))
              .reset_index().melt('country',var_name='Barrier',value_name='Count'))
    fig=px.bar(bar_df,x='Barrier',y='Count',color='country',barmode='group',height=350)
    st.plotly_chart(fig,use_container_width=True)

# ===================== TAB 2 Classification ==============================
with tabs[1]:
    st.header("Purchase-Intent Classification")
    y=df['willing_to_purchase_12m'];X=dummies(df.drop(columns=['willing_to_purchase_12m']))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    scaler=StandardScaler()
    Xtr_s=scaler.fit_transform(Xtr.select_dtypes('number'))
    Xte_s=scaler.transform(Xte.select_dtypes('number'))
    models={
        "KNN":KNeighborsClassifier(7),
        "Decision Tree":DecisionTreeClassifier(max_depth=8,random_state=42),
        "Random Forest":RandomForestClassifier(200,random_state=42),
        "GBRT":GradientBoostingClassifier(random_state=42)}
    metrics,roc_dict={},{}
    for n,m in models.items():
        if n=="KNN":
            m.fit(Xtr_s,ytr);pred=m.predict(Xte_s);pro=m.predict_proba(Xte_s)
            pred_tr=m.predict(Xtr_s)
        else:
            m.fit(Xtr,ytr);pred=m.predict(Xte);pro=m.predict_proba(Xte);pred_tr=m.predict(Xtr)
        metrics[n]={"Train":accuracy_score(ytr,pred_tr),"Test":accuracy_score(yte,pred),
                    "Prec":precision_score(yte,pred,average='weighted'),
                    "Rec":recall_score(yte,pred,average='weighted'),
                    "F1":f1_score(yte,pred,average='weighted')}
        fpr,tpr,_=roc_curve(label_binarize(yte,classes=[0,1,2]).ravel(),pro.ravel())
        roc_dict[n]=(fpr,tpr)
    st.dataframe(pd.DataFrame(metrics).T.style.format("{:.2f}"))

    # Feature importance bar (RF)
    st.subheader("Feature Importance – Random Forest")
    imp=pd.Series(models["Random Forest"].feature_importances_,index=X.columns).nlargest(15)
    st.bar_chart(imp)

    # SHAP summary toggle
    with st.expander("Show SHAP summary"):
        explainer=shap.TreeExplainer(models["Random Forest"]);sh=explainer.shap_values(Xte.iloc[:300])
        fig=shap.summary_plot(sh,features=Xte.iloc[:300],show=False)
        st.pyplot(bbox_inches='tight')

    cmc,roc_c=st.columns(2)
    with cmc:
        st.markdown("##### Confusion Matrix")
        alg=st.selectbox("Model:",list(models.keys()),key="cm_select")
        mdl=models[alg];pred_cm=mdl.predict(Xte_s if alg=="KNN" else Xte)
        pretty_cm(confusion_matrix(yte,pred_cm),["No","Maybe","Yes"])
    with roc_c:
        st.markdown("##### ROC & Lift")
        multi_roc(roc_dict)
        skplt.metrics.plot_cumulative_gain(yte,models["Random Forest"].predict_proba(Xte))
        st.pyplot(plt.gcf(),clear_figure=True,use_container_width=False)

# ===================== TAB 3 Clustering ==================================
with tabs[2]:
    st.header("Segmentation (K-means)")
    num=num_df(df);scaled=StandardScaler().fit_transform(num)
    inertia=[KMeans(k,n_init='auto',random_state=42).fit(scaled).inertia_ for k in range(2,11)]
    st.line_chart(pd.Series(inertia,index=range(2,11)),use_container_width=True)
    ksel=st.slider("Clusters",2,10,4)
    km=KMeans(ksel,n_init='auto',random_state=42).fit(scaled);df['cluster']=km.labels_
    centers=pd.DataFrame(km.cluster_centers_,columns=num.columns)
    st.dataframe(centers)

    # Parallel coords
    st.subheader("Cluster Profiles – Parallel Coordinates")
    fig=px.parallel_coordinates(centers,color=centers.index,
                                color_continuous_scale=px.colors.sequential.Greens)
    st.plotly_chart(fig,use_container_width=True)

    # Radar chart
    st.subheader("Cluster Radar")
    rad=centers.copy();rad['label']=rad.index
    fig=go.Figure()
    for i,row in rad.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[:-1].tolist(),theta=centers.columns,fill='toself',name=f'Cluster {i}'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[rad.min().min(),rad.max().max()])),
                      height=450,showlegend=True)
    st.plotly_chart(fig,use_container_width=True)

# ===================== TAB 4 Association Rules ============================
with tabs[3]:
    st.header("Apriori Rules")
    hot=[c for c in df.columns if any(p in c for p in ("own_","reason_","barrier_","pref_","src_"))]
    sel=st.multiselect("Columns",hot,default=hot[:20])
    msup=st.slider("Min support",0.01,0.5,0.05,0.01)
    mconf=st.slider("Min confidence",0.1,1.0,0.3,0.05)
    if st.button("Run Apriori"):
        basket=df[sel].astype(bool);freq=apriori(basket,min_support=msup,use_colnames=True)
        rules=association_rules(freq,metric="confidence",min_threshold=mconf)
        rules["antecedents"]=rules["antecedents"].apply(lambda x:", ".join(x))
        rules["consequents"]=rules["consequents"].apply(lambda x:", ".join(x))
        st.dataframe(rules.head(10)[["antecedents","consequents","support","confidence","lift"]])
        if st.checkbox("Show rule network"):
            plot_network(rules.head(15))

# ===================== TAB 5 Regression ===================================
with tabs[4]:
    st.header("WTP Prediction")
    yreg=df["max_willingness_to_pay_usd"];Xreg=dummies(df.drop(columns=["max_willingness_to_pay_usd","cluster"]))
    Xtr,Xte,ytr,yte=train_test_split(Xreg,yreg,test_size=.2,random_state=42)
    regs={"Linear":LinearRegression(),"Ridge":Ridge(),"Lasso":Lasso(alpha=.001),
          "DT":DecisionTreeRegressor(max_depth=6,random_state=42)}
    out={}
    for n,m in regs.items():
        m.fit(Xtr,ytr);pred=m.predict(Xte)
        out[n]={"R²":r2_score(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred))}
    st.dataframe(pd.DataFrame(out).T.style.format("{:.2f}"))

    # Waterfall explanation
    st.subheader("Explain One Prediction (SHAP Waterfall)")
    row=st.number_input("Row index",0,len(Xte)-1,0,int_step=1)
    expl=shap.TreeExplainer(regs["Ridge"]);val=expl.shap_values(Xte.iloc[row:row+1])
    shap_fig=shap.plots._waterfall.waterfall_legacy(expl.expected_value,val[0],
                                                    feature_names=Xte.columns,show=False)
    st.pyplot(bbox_inches='tight')

# ===================== TAB 6 Advanced =====================================
with tabs[5]:
    st.header("Advanced Insights")
    st.subheader("Sankey: Environmental Concern ➜ Purchase Intent")
    concern_bins=pd.cut(df['env_concern_score'],bins=[0,2,4,5],labels=["Low","Medium","High"])
    sank=pd.crosstab(concern_bins,df['willing_to_purchase_12m'])
    src,tgt,vals=[],[],[]
    for i,c in enumerate(sank.index):
        for j,cls in enumerate(sank.columns):
            src.append(i);tgt.append(len(sank.index)+j);vals.append(sank.loc[c,cls])
    label=list(sank.index)+['No','Maybe','Yes']
    fig=go.Figure(data=[go.Sankey(node=dict(label=label,pad=15),
        link=dict(source=src,target=tgt,value=vals))])
    fig.update_layout(height=400,margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig,use_container_width=True)
