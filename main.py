import streamlit as st
import joblib
import pandas as pd
import numpy as np


# Load the trained model
model = joblib.load('model/xgboost_pipeline.pkl')

# Load the dataset (optional: for displaying data or mapping values)
df = joblib.load("model/df.pkl")


st.title("Laptop Price Prediction App 💻")

st.write("Enter the details below to predict the price:")

# Collect user input
company = st.selectbox("Company", df['Company'].unique())
typename = st.selectbox("Type Name", df['TypeName'].unique())
ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.radio("Touchscreen", ["No", "Yes"])
ips = st.radio("IPS Display", ["No", "Yes"])
ppi = st.number_input("PPI (Pixel Density)", min_value=50.0, max_value=300.0, step=1.0)
cpu_brand = st.selectbox("CPU Brand", df['Cpu brand'].unique())
hdd = st.selectbox("HDD (GB)", sorted(df['HDD'].unique()))
ssd = st.selectbox("SSD (GB)", sorted(df['SSD'].unique()))
gpu_brand = st.selectbox("GPU Brand", df['Gpu brand'].unique())
os = st.selectbox("Operating System", df['Os'].unique())

# Convert categorical inputs
touchscreen = 1 if touchscreen == "Yes" else 0
ips = 1 if ips == "Yes" else 0

# Create input dataframe
input_data = pd.DataFrame([[company, typename, ram, weight, touchscreen, ips, ppi, cpu_brand, hdd, ssd, gpu_brand, os]],
                          columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'Ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'Os'])

# Make prediction
if st.button("Predict Price 💰"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Predicted Laptop Price: ₹{predicted_price}")
