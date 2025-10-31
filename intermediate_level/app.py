import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Load model and scaler ---
model_path = os.path.join(os.path.dirname(__file__), "model_car.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler_car.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")
st.title("🚗 Car Selling Price Prediction")
st.markdown("Predict the approximate **selling price** of your car based on its features.")

# --- Sidebar inputs ---
st.sidebar.header("Enter Car Details")

Present_Price = st.sidebar.number_input("Showroom Price (in lakhs)", 0.0, 50.0, 5.0)
Kms_Driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 30000)
Owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])
Fuel_Type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
Seller_Type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
Transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
Years_old = st.sidebar.number_input("Years of Service", 0, 30, 5)

# --- Encode categorical inputs ---
Fuel_Type_map = {"Petrol": 2, "Diesel": 0, "CNG": 1}
Seller_Type_map = {"Dealer": 0, "Individual": 1}
Transmission_map = {"Manual": 1, "Automatic": 0}

data = pd.DataFrame([{
    "Present_Price": Present_Price,
    "Kms_Driven": Kms_Driven,
    "Owner": Owner,
    "Fuel_Type": Fuel_Type_map[Fuel_Type],
    "Seller_Type": Seller_Type_map[Seller_Type],
    "Transmission": Transmission_map[Transmission],
    "Years_old": Years_old
}])

# --- Prediction ---
if st.button("🔍 Predict Selling Price"):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} lakhs")

