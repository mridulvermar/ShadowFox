import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

# Load model and scaler
with open(os.path.join(BASE_DIR, "model_car.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler_car.pkl"), "rb") as f:
    scaler = pickle.load(f)

st.title("🚗 Car Price Prediction App")

# User inputs
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
present_price = st.number_input("Showroom Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Encode categorical variables
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

# Calculate car age
current_year = 2020  # 👈 use the same base year as used in training
age = current_year - year

# Order of features must match the one used during training
input_features = np.array([[present_price,
                            kms_driven,
                            owner,
                            age,
                            fuel_map[fuel_type],
                            seller_map[seller_type],
                            trans_map[transmission]]])

# Prediction
if st.button("Predict Price 💰"):
    try:
        scaled_data = scaler.transform(input_features)
        predicted_price = model.predict(scaled_data)[0]
        st.success(f"Estimated Selling Price: ₹ {predicted_price:.2f} lakhs")
    except ValueError as e:
        st.error("⚠️ Feature mismatch — please verify the feature order and retrain the scaler.")
        st.text(str(e))
