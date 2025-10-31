import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model_car.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🚗 Car Selling Price Prediction App")

# Collect user input
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025)

if st.button("Predict Price"):
    years_old = 2025 - year

    # Match exact training columns
    input_df = pd.DataFrame([{
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Fuel_Type': 2 if fuel_type == 'Petrol' else 1 if fuel_type == 'Diesel' else 0,
        'Seller_Type': 1 if seller_type == 'Individual' else 0,
        'Transmission': 1 if transmission == 'Manual' else 0,
        'Owner': owner,
        'Years_old': years_old
    }])

    # Apply scaler with same feature order as during training
    feature_order = ['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Years_old']
    input_df = input_df[feature_order]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Estimated Selling Price: ₹ {round(prediction[0], 2)} lakhs")
