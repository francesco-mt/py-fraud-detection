"""
Credit Card Fraud Detection — Streamlit Demo App
Portfolio project for Data Analyst roles in Malta (Fintech / iGaming)

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | Fintech ML Demo",
    page_icon="💳",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    """
    **Portfolio Project** — ML-powered fraud detection aligned with MFSA AML monitoring requirements.  
    Built with XGBoost + SHAP explainability for compliance-ready predictions.
    """
)
st.divider()

# ── Sidebar: Model info ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this model")
    st.markdown("""
    **Dataset:** 284,807 real European card transactions  
    **Fraud rate:** ~0.17%  
    **Model:** XGBoost (best AUC-ROC)  
    **Imbalance handling:** SMOTE  
    **Explainability:** SHAP values
    
    ---
    **Metrics on test set:**
    - AUC-ROC: ~0.98
    - Precision (fraud): ~0.90
    - Recall (fraud): ~0.82
    
    ---
    > Threshold can be tuned to balance fraud capture rate vs false positive rate.
    """)

    threshold = st.slider(
        "🎚️ Decision Threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Lower = catch more fraud (more false positives). Higher = fewer false positives (miss more fraud)."
    )

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('src/xgb_fraud_model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found at src/xgb_fraud_model.pkl. Train the notebook first.")
        return None

# ── Load dataset for sampling ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = [
        'data/creditcard.csv',
        '../data/creditcard.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

model = load_model()
df    = load_data()

# ── Feature columns in training order ─────────────────────────────────────────
FEATURE_COLS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                'V28', 'Amount']

# ── Session state: store the currently loaded transaction ──────────────────────
if 'current_tx' not in st.session_state:
    st.session_state.current_tx = {col: 0.0 for col in FEATURE_COLS}
    st.session_state.tx_label   = None

# ── Transaction loader buttons ─────────────────────────────────────────────────
st.subheader("🎲 Load a Sample Transaction")

if df is not None:
    col_a, col_b, col_c = st.columns(3)

    def _apply_tx(values: dict, label):
        """Write transaction values into both current_tx and each widget's key."""
        st.session_state.current_tx = values
        st.session_state.tx_label   = label
        st.session_state['inp_amount'] = values['Amount']
        st.session_state['inp_time']   = values['Time']
        for i in range(1, 29):
            st.session_state[f'v{i}'] = values[f'V{i}']

    with col_a:
        if st.button("🚨 Load Random FRAUD Transaction", use_container_width=True):
            fraud_df = df[df['Class'] == 1]
            row = fraud_df.iloc[np.random.randint(0, len(fraud_df))]
            _apply_tx({col: float(row[col]) for col in FEATURE_COLS}, "fraud")

    with col_b:
        if st.button("✅ Load Random LEGITIMATE Transaction", use_container_width=True):
            legit_df = df[df['Class'] == 0]
            row = legit_df.iloc[np.random.randint(0, len(legit_df))]
            _apply_tx({col: float(row[col]) for col in FEATURE_COLS}, "legit")

    with col_c:
        if st.button("🔄 Reset to Zeros", use_container_width=True):
            _apply_tx({col: 0.0 for col in FEATURE_COLS}, None)

    if st.session_state.tx_label == "fraud":
        st.info("🚨 A known **fraudulent** transaction has been loaded. See if the model catches it!")
    elif st.session_state.tx_label == "legit":
        st.info("✅ A known **legitimate** transaction has been loaded.")
else:
    st.warning("Dataset not found — random sampling unavailable. Place creditcard.csv in the data/ folder.")

st.divider()

# ── Transaction input fields ───────────────────────────────────────────────────
st.subheader("🔍 Transaction Details")
st.caption("Values are auto-filled when you load a sample above. You can also edit them manually.")

tx = st.session_state.current_tx

col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Amount (€)",  value=tx['Amount'], format="%.4f", key="inp_amount")
    time   = st.number_input("Time (secs)", value=tx['Time'],   format="%.4f", key="inp_time")

with col2:
    st.markdown("**PCA Features V1–V14**")
    v1_14 = {f"V{i}": st.number_input(f"V{i}", value=tx[f"V{i}"], format="%.4f", key=f"v{i}")
             for i in range(1, 15)}

with col3:
    st.markdown("**PCA Features V15–V28**")
    v15_28 = {f"V{i}": st.number_input(f"V{i}", value=tx[f"V{i}"], format="%.4f", key=f"v{i}")
              for i in range(15, 29)}

# ── Analyse button ─────────────────────────────────────────────────────────────
if st.button("🚨 Analyse Transaction", type="primary"):
    if model:
        input_data = pd.DataFrame([{
            'Time': time, **v1_14, **v15_28, 'Amount': amount
        }])

        # Ensure correct column order
        input_data = input_data[FEATURE_COLS]

        fraud_prob = model.predict_proba(input_data)[0][1]
        is_fraud   = fraud_prob >= threshold

        st.divider()
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if is_fraud:
                st.error(f"🚨 **FLAGGED AS FRAUD**\n\nFraud Probability: **{fraud_prob:.1%}**")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**\n\nFraud Probability: **{fraud_prob:.1%}**")

            st.metric("Threshold used", f"{threshold:.2f}")

            if st.session_state.tx_label:
                actual = "FRAUD" if st.session_state.tx_label == "fraud" else "LEGITIMATE"
                predicted = "FRAUD" if is_fraud else "LEGITIMATE"
                if actual == predicted:
                    st.success(f"✅ Correct! Actual label: **{actual}**")
                else:
                    st.warning(f"❌ Missed! Actual label: **{actual}** — model predicted **{predicted}**")

        with col_res2:
            try:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)

                fig, ax = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=input_data.iloc[0],
                        feature_names=input_data.columns.tolist()
                    ),
                    show=False
                )
                st.pyplot(fig)
                st.caption("SHAP waterfall: features pushing prediction towards fraud (red) or away (blue)")
            except Exception as e:
                st.info(f"SHAP plot unavailable: {e}")
    else:
        st.error("Model not loaded. Please train the notebook first.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Portfolio project | Credit Card Fraud Detection | Built with Python, XGBoost, SHAP & Streamlit")