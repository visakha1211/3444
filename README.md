# EcoWise Market Feasibility Dashboard

Interactive Streamlit web‑app for exploring the synthetic *EcoWise Appliances* dataset,
running machine‑learning pipelines, and mining association rules.

## 1. Quick start on Streamlit Cloud

1. **Fork / clone** this repo (contains `app.py`, data, and requirements).
2. Push to your GitHub account (public or private).
3. Go to **https://share.streamlit.io** → *New app*  
   • Select repository & branch  
   • Main file: `app.py`
4. Click **Deploy**. Streamlit Cloud will install packages from `requirements.txt`
   and start your dashboard.

> **Tip :** The sample dataset `ecowise_full_arm_ready.csv` is already included.
> Upload your own CSVs via the sidebar uploader.

## 2. Folder layout

```
├── app.py                 # Streamlit multi‑tab app
├── requirements.txt       # Python libs
├── README.md              # this file
└── ecowise_full_arm_ready.csv   # sample data (1 000 rows)
```

## 3. Key features

* **Data Visualisation** – 10+ charts, dynamic filters.
* **Classification** – KNN, DT, RF, GBRT with metrics table, ROC, confusion matrix,
  and prediction batch‑upload / download.
* **Clustering** – K‑means with elbow plot, interactive k‑slider, persona table,
  downloadable labels.
* **Association‑Rule Mining** – Apriori with adjustable thresholds and top‑10 rules table.
* **Regression** – Linear, Ridge, Lasso, DT regressors for quick insights.

All plots include short interpretations to help non‑technical users.
