#  Credit Card Fraud Detection
### Fintech ML Portfolio Project — Data Analyst 

> Built to demonstrate ML skills relevant to MFSA-regulated fintech companies in Malta.  
> Aligned with AML monitoring requirements and compliance-ready explainability.

---

##  Objective

Build a binary classifier to detect fraudulent credit card transactions, with explainable predictions suitable for compliance reporting.

---

##  Project Structure

```
fraud_detection/
├── data/
│   └── creditcard.csv          ← Download from Kaggle (link below)
├── notebooks/
│   └── fraud_detection.ipynb   ← Main analysis notebook (start here)
├── src/
│   └── xgb_fraud_model.pkl     ← Saved model (generated after running notebook)
├── app/
│   └── app.py                  ← Streamlit demo app
├── outputs/
│   ├── class_distribution.png
│   ├── roc_comparison.png
│   ├── shap_importance.png
│   └── ...
└── README.md
```

---

##  Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 anonymised European cardholder transactions
- Fraud rate: ~0.17% (highly imbalanced)
- Features: V1–V28 (PCA-transformed), Amount, Time
- Target: Class (0 = Legitimate, 1 = Fraud)

**Download steps:**
1. Create a free Kaggle account
2. Download `creditcard.csv`
3. Place it in the `data/` folder

---

##  Setup

#### Prerequisites

Python 3.8+ — python.org
pip — comes bundled with Python
Jupyter Notebook or JupyterLab — to run the analysis notebook
A Kaggle account (free) — to download the dataset

#### 1. Clone the Repository

```bash
git clone https://github.com/francesco-mt/py-fraud-detection.git
cd py-fraud-detection
```

#### 2. Install Dependencies

```bash
pip install requirements.txt
```

#### 3. Download the dataset
use the link provided at the top of the notebook and save the file creditcard.csv in the 'data' folder

#### 4. Run the notebook

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

#### 5. Run the Streamlit app

```bash
cd app && streamlit run app.py
```

---

##  Machine Learning Pipeline

| Step | Detail |
|------|--------|
| **EDA** | Class imbalance analysis, amount/time distributions, fraud patterns |
| **Preprocessing** | StandardScaler on Amount & Time; stratified train/test split |
| **Imbalance handling** | SMOTE applied to training set only |
| **Models** | Logistic Regression (baseline) → Random Forest → XGBoost |
| **Evaluation** | AUC-ROC, Precision, Recall, F1 |
| **Explainability** | SHAP TreeExplainer — global + per-transaction explanations |
| **Threshold tuning** | Precision/Recall/F1 curve to select business-optimal cutoff |

---

##  Results

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | ~0.97 |
| Random Forest | ~0.98 |
| **XGBoost** | **~0.98** |

---

##  Key Design Decisions

**Why not use accuracy?**  
With only 0.17% fraud, a model predicting "always legitimate" achieves 99.83% accuracy — yet catches zero fraud. AUC-ROC, Precision, and Recall meet the project's needs better.

**Why SHAP?**  
To show why a transaction was flagged, according to AML compliance obligations Malta's MFSA-regulated fintechs are subject to.

**Why threshold tuning?**  
To accomodate for various business goals.

---


##  Regulatory Context

This project is designed with Malta's regulatory landscape in mind:

- **MFSA** (Malta Financial Services Authority) — enforces AML/CFT obligations for licensed payment institutions
- **FIAU** (Financial Intelligence Analysis Unit) — requires suspicious transaction reporting
- **PSD2** — mandates strong customer authentication and fraud monitoring for payment processors
