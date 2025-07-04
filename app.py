# ──────────────────────────────────────────────────────────────────────────
#  EcoWise Insight Studio – 2025  (complete, polished version)
#  Put a 1200×180 PNG hero image named banner_ecowise.png next to this file
# ──────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from base64 import b64encode
import plotly.express as px
import plotly.graph_objects as go
import scikitplot as skplt
import shap
import networkx as nx
from pyvis.network import Network
import io, tempfile, json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ════════════════  PAGE CONFIG & THEME  ═══════════════════════════════════
st.set_page_config(page_title="EcoWise Insight Studio", layout="wide")

# ── inline banner (if present) ────────────────────────────────────────────
banner = Path("banner_ecowise.png")
if banner.exists():
    b64 = b64encode(banner.read_bytes()).decode()
    st.markdown(
        f'<div style="width:100%;text-align:center">'
        f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;"></div>',
        unsafe_allow_html=True
    )

# ── emerald / teal CSS tweaks ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        :root { --primary:#2ecc71; --accent:#1abc9c; --bg:#f7fdf9; --txt:#033e26; }
        html,body,[class*="View"]{background:var(--bg)!important;color:var(--txt);}
        h1,h2,h3,h4{color:var(--primary);}
        button[kind="primary"]{background:var(--accent)!important;border-radius:8px;}
        button[kind="primary"]:hover{background:#17a689!important;}
        div[data-testid="metric-container"]{
            background:#fffffff2;border:1px solid #e5f7ed;border-radius:12px;
            box-shadow:0 1px 3px rgba(0,0,0,.05);
        }
        section[data-testid="stSidebar"]{background:#e9f9f1;border-right:2px solid #c8efd8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 EcoWise Market Feasibility Dashboard")

# ════════════════  DATA  ══════════════════════════════════════════════════
csv_file = st.sidebar.file_uploader("Upload CSV", type="csv")

@st.cache_data
def load_data(f):
    return pd.read_csv(f) if f else pd.read_csv("ecowise_full_arm_ready.csv")

df = load_data(csv_file)
st.sidebar.success(f"Loaded **{df.shape[0]} rows × {df.shape[1]} cols**")

# ════════════════  HELPER FUNCTIONS  ══════════════════════════════════════
def num_df(d): return d.select_dtypes("number")
def dummies(d): return pd.get_dummies(d, drop_first=True)

def tiny_cm(cm, labels):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6); ax.set_yticklabels(labels, fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=6)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=False)

def tiny_roc(curves):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for n, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{n} (AUC {auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],"--",lw=1,color="#888")
    ax.set_xlabel("FPR", fontsize=7); ax.set_ylabel("TPR", fontsize=7)
    ax.legend(frameon=False, fontsize=6)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=False)

def rule_network(rules, n=15):
    nt = Network(height="460px", width="100%", bgcolor="#ffffff")
    for _, row in rules.head(n).iterrows():
        src = row["antecedents"]; dst = row["consequents"]
        nt.add_node(src, src, title=src, color="#1abc9c")
        nt.add_node(dst, dst, title=dst, color="#27ae60")
        nt.add_edge(src, dst, value=row["lift"])
    tmp = Path(tempfile.mkdtemp()) / "net.html"
    nt.show(str(tmp))
    st.components.v1.html(tmp.read_text(), height=480, scrolling=False)

# Matplotlib colour cycle
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=["#1abc9c", "#16a085", "#2ecc71", "#27ae60"])
})

# ════════════════  TABS  ══════════════════════════════════════════════════
tabs = st.tabs(
    ["📊 Visuals", "🤖 Classify", "📍 Cluster", "🔗 Rules", "📈 Regression", "✨ Advanced"]
)

# -------------------------------------------------------------------------
# TAB 1 – VISUALS
# -------------------------------------------------------------------------
with tabs[0]:
    st.header("Descriptive Insights")
    col1, col2 = st.columns(2)

    # Correlation heatmap
    with col1:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(num_df(df).corr(), cmap="viridis")
        ax.set_xticks(range(len(num_df(df).columns)))
        ax.set_xticklabels(num_df(df).columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(num_df(df).columns)))
        ax.set_yticklabels(num_df(df).columns, fontsize=6)
        fig.colorbar(im, fraction=0.035); st.pyplot(fig, use_container_width=True)

    # Income histogram
    with col2:
        st.subheader("Income by Country")
        fig = px.histogram(df, x="household_income_usd", color="country",
                           nbins=25, opacity=0.6)
        fig.update_layout(height=350, xaxis_range=[105,
                        df["household_income_usd"].quantile(0.95)])
        st.plotly_chart(fig, use_container_width=True)

    # KPI metrics & WTP box
    with col1:
        st.metric("Avg. Monthly Bill", f"${df['monthly_energy_bill_usd'].mean():.0f}")
        st.metric("Median Max WTP",     f"${df['max_willingness_to_pay_usd'].median():.0f}")
        pct = (df["willing_to_purchase_12m"] > 0).mean() * 100
        st.metric("Intent ≥ 'Maybe'", f"{pct:.1f}%")

    with col2:
        st.subheader("WTP vs Environmental Concern")
        fig = px.box(df, x="env_concern_score", y="max_willingness_to_pay_usd",
                     height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Average WTP by Country (Choropleth)")
    iso = {"India": "IND", "UAE": "ARE", "Singapore": "SGP"}
    map_df = df.groupby("country")["max_willingness_to_pay_usd"].mean().reset_index()
    map_df["iso"] = map_df["country"].map(iso)
    fig = px.choropleth(map_df, locations="iso", color="max_willingness_to_pay_usd",
                        color_continuous_scale="greens", hover_name="country",
                        labels={"max_willingness_to_pay_usd": "Avg WTP (USD)"})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Barrier Frequency by Country")
    b_cols = [c for c in df.columns if c.startswith("barrier_")]
    bar_df = (df.groupby("country")[b_cols].sum()
                .rename(columns=lambda x: x.replace("barrier_", ""))
                .reset_index()
                .melt("country", var_name="Barrier", value_name="Count"))
    fig = px.bar(bar_df, x="Barrier", y="Count", color="country",
                 barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 2 – CLASSIFICATION
# -------------------------------------------------------------------------
with tabs[1]:
    st.header("Purchase-Intent Classification")

    y = df["willing_to_purchase_12m"]
    X = dummies(df.drop(columns=["willing_to_purchase_12m"]))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.select_dtypes("number"))
    X_te_s = scaler.transform(X_te.select_dtypes("number"))

    models = {
        "KNN": KNeighborsClassifier(7),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(200, random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42),
    }

    perf, roc_curves = {}, {}
    for name, mdl in models.items():
        if name == "KNN":
            mdl.fit(X_tr_s, y_tr)
            preds = mdl.predict(X_te_s)
            probs = mdl.predict_proba(X_te_s)
            preds_tr = mdl.predict(X_tr_s)
        else:
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
            probs = mdl.predict_proba(X_te)
            preds_tr = mdl.predict(X_tr)

        perf[name] = {
            "Train": accuracy_score(y_tr, preds_tr),
            "Test":  accuracy_score(y_te, preds),
            "Prec":  precision_score(y_te, preds, average="weighted"),
            "Rec":   recall_score(y_te, preds, average="weighted"),
            "F1":    f1_score(y_te, preds, average="weighted"),
        }

        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
        roc_curves[name] = (fpr, tpr)

    st.subheader("Performance Grid")
    st.dataframe(pd.DataFrame(perf).T.style.format("{:.2f}"))

    # Feature importance
    st.subheader("Feature Importance – Random Forest")
    top = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).nlargest(15)
    st.bar_chart(top)

    # SHAP summary (optional)
    with st.expander("Show SHAP Summary"):
        explainer = shap.TreeExplainer(models["Random Forest"])
        shap_vals = explainer.shap_values(X_te.iloc[:300])
        shap.summary_plot(shap_vals, X_te.iloc[:300], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

    cm_col, roc_col = st.columns(2)
    with cm_col:
        st.markdown("##### Confusion Matrix")
        which = st.selectbox("Model:", list(models.keys()), key="cm_model")
        m = models[which]
        preds_cm = m.predict(X_te_s if which == "KNN" else X_te)
        tiny_cm(confusion_matrix(y_te, preds_cm), ["No", "Maybe", "Yes"])

    with roc_col:
        st.markdown("##### ROC & Lift")
        tiny_roc(roc_curves)
        skplt.metrics.plot_cumulative_gain(
            y_te, models["Random Forest"].predict_proba(X_te))
        st.pyplot(plt.gcf(), clear_figure=True, use_container_width=False)

# -------------------------------------------------------------------------
# TAB 3 – CLUSTERING
# -------------------------------------------------------------------------
with tabs[2]:
    st.header("Segmentation (K-means)")
    num = num_df(df)
    scaled = StandardScaler().fit_transform(num)
    inertia = [KMeans(k, n_init="auto", random_state=42).fit(scaled).inertia_
               for k in range(2, 11)]
    st.line_chart(pd.Series(inertia, index=range(2, 11)))

    k_val = st.slider("Clusters", 2, 10, 4)
    km = KMeans(k_val, n_init="auto", random_state=42).fit(scaled)
    df["cluster"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=num.columns)
    st.dataframe(centers)

    st.subheader("Parallel Coordinates")
    fig = px.parallel_coordinates(
        centers, color=centers.index,
        color_continuous_scale=px.colors.sequential.Greens
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Radar Profile")
    fig = go.Figure()
    for i, row in centers.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row.tolist(), theta=centers.columns,
            fill="toself", name=f"Cluster {i}"))
    fig.update_layout(height=430, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 4 – ASSOCIATION RULES
# -------------------------------------------------------------------------
with tabs[3]:
    st.header("Apriori Rules")
    oh_cols = [c for c in df.columns if any(p in c for p in
               ("own_", "reason_", "barrier_", "pref_", "src_"))]
    chosen = st.multiselect("Columns", oh_cols, default=oh_cols[:20])
    min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)

    if st.button("Run Apriori"):
        basket = df[chosen].astype(bool)
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(s))
        rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(s))

        st.dataframe(rules.sort_values("lift", ascending=False)
                     .head(10)[["antecedents", "consequents",
                                "support", "confidence", "lift"]])

        if st.checkbox("Show rule network"):
            rule_network(rules)

# -------------------------------------------------------------------------
# TAB 5 – REGRESSION
# -------------------------------------------------------------------------
with tabs[4]:
    st.header("Spend Prediction (Regression)")
    y_reg = df["max_willingness_to_pay_usd"]
    X_reg = dummies(df.drop(columns=["max_willingness_to_pay_usd", "cluster"]))
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.001),
        "DT Reg": DecisionTreeRegressor(max_depth=6, random_state=42),
    }
    res = {}
    for n, m in reg_models.items():
        m.fit(X_tr, y_tr); pred = m.predict(X_te)
        res[n] = {"R²": r2_score(y_te, pred), "RMSE": np.sqrt(mean_squared_error(y_te, pred))}
    st.dataframe(pd.DataFrame(res).T.style.format("{:.2f}"))

    st.subheader("Explain One Prediction (SHAP Waterfall)")
    idx = st.number_input("Row index", 0, len(X_te) - 1, 0, step=1)
    expl = shap.TreeExplainer(reg_models["Ridge"])
    shap_vals = expl.shap_values(X_te.iloc[idx:idx+1])
    shap.plots._waterfall.waterfall_legacy(
        expl.expected_value, shap_vals[0], feature_names=X_te.columns, show=False
    )
    st.pyplot(plt.gcf(), clear_figure=True)

# -------------------------------------------------------------------------
# TAB 6 – ADVANCED
# -------------------------------------------------------------------------
with tabs[5]:
    st.header("Advanced Insights")
    st.subheader("Sankey: Concern ➜ Intent")
    bins = pd.cut(df["env_concern_score"], [0, 2, 4, 5],
                  labels=["Low", "Medium", "High"])
    sankey = pd.crosstab(bins, df["willing_to_purchase_12m"])
    src, tgt, val = [], [], []
    for i, con in enumerate(sankey.index):
        for j, cls in enumerate(sankey.columns):
            src.append(i)
            tgt.append(len(sankey.index) + j)
            val.append(sankey.loc[con, cls])
    labels = list(sankey.index) + ["No", "Maybe", "Yes"]
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15),
        link=dict(source=src, target=tgt, value=val)
    ))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
