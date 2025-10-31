import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load models and scaler
lr = pickle.load(open("model_lr.pkl", "rb"))
dt = pickle.load(open("model_dt.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.markdown("### Predict Boston Housing Prices using Linear Regression and Decision Tree Models")

# Sidebar inputs
st.sidebar.header("Enter House Features")
CRIM = st.sidebar.number_input("CRIM (per capita crime rate)", 0.0, 100.0, 0.1)
ZN = st.sidebar.number_input("ZN (residential land zoned)", 0.0, 100.0, 18.0)
INDUS = st.sidebar.number_input("INDUS (non-retail acres per town)", 0.0, 30.0, 2.31)
CHAS = st.sidebar.selectbox("CHAS (Charles River dummy variable)", [0, 1])
NOX = st.sidebar.number_input("NOX (nitric oxides concentration)", 0.0, 1.0, 0.538)
RM = st.sidebar.number_input("RM (avg rooms per dwelling)", 1.0, 10.0, 6.575)
AGE = st.sidebar.number_input("AGE (proportion of old units)", 0.0, 100.0, 65.2)
DIS = st.sidebar.number_input("DIS (distance to employment centers)", 0.0, 15.0, 4.09)
RAD = st.sidebar.number_input("RAD (accessibility to highways)", 1, 24, 1)
TAX = st.sidebar.number_input("TAX (property tax rate)", 0.0, 1000.0, 296.0)
PTRATIO = st.sidebar.number_input("PTRATIO (pupil-teacher ratio)", 5.0, 30.0, 15.3)
B = st.sidebar.number_input("B (black population proportion)", 0.0, 400.0, 396.9)
LSTAT = st.sidebar.number_input("LSTAT (lower status %)", 0.0, 40.0, 4.98)

# Create dataframe for input
new_data = pd.DataFrame([{
    'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'CHAS': CHAS, 'NOX': NOX,
    'RM': RM, 'AGE': AGE, 'DIS': DIS, 'RAD': RAD, 'TAX': TAX,
    'PTRATIO': PTRATIO, 'B': B, 'LSTAT': LSTAT
}])

# Prediction
if st.button("🔍 Predict House Price"):
    new_data_scaled = scaler.transform(new_data)
    lr_pred = lr.predict(new_data_scaled)[0]
    dt_pred = dt.predict(new_data)[0]

    st.subheader("📈 Prediction Results")
    st.success(f"**Linear Regression:** ${lr_pred*1000:.2f}")
    st.info(f"**Decision Tree:** ${dt_pred*1000:.2f}")

    st.caption("*(Predictions are in thousands of dollars)*")
