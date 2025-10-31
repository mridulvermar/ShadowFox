import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load trained model and scaler
# -------------------------------
with open("model_car.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="🚗 Car Selling Price Predictor", layout="centered")
st.title("🚗 Car Selling Price Prediction App")
st.markdown("### Get an estimated selling price for your car")

# -------------------------------
# Sidebar for Inputs
# -------------------------------
st.sidebar.header("🔧 Input Car Details")

Year = st.sidebar.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2018)
Present_Price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0, step=0.1)
Kms_Driven = st.sidebar.number_input("Kilometers Driven", min_value=0, value=20000, step=500)
Owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])
Fuel_Type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
Seller_Type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
Transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# -------------------------------
# Preprocess Inputs
# -------------------------------
# Years used
Years_old = 2025 - Year

# Encode categorical variables
Fuel_Type_Petrol = 1 if Fuel_Type == "Petrol" else 0
Fuel_Type_Diesel = 1 if Fuel_Type == "Diesel" else 0
Fuel_Type_CNG = 1 if Fuel_Type == "CNG" else 0

Seller_Type_Individual = 1 if Seller_Type == "Individual" else 0
Transmission_Manual = 1 if Transmission == "Manual" else 0

# Create DataFrame
data = pd.DataFrame({
    'Present_Price': [Present_Price],
    'Kms_Driven': [Kms_Driven],
    'Owner': [Owner],
    'Years_old': [Years_old],
    'Fuel_Type_Diesel': [Fuel_Type_Diesel],
    'Fuel_Type_Petrol': [Fuel_Type_Petrol],
    'Fuel_Type_CNG': [Fuel_Type_CNG],
    'Seller_Type_Individual': [Seller_Type_Individual],
    'Transmission_Manual': [Transmission_Manual]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Selling Price"):
    # Scale numeric features
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    st.subheader("💰 Predicted Selling Price")
    st.success(f"Estimated Price: ₹ {prediction:.2f} lakhs")

    st.caption("*(Prediction is based on your input values)*")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-learn")
