import streamlit as st
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "backend", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "..", "backend", "scaler.pkl")

try:
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    st.write("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("Health Anomaly Detection")

st.write("Enter patient vitals:")

# Inputs
hr = st.number_input("Heart Rate (bpm)", 30, 200, 80)
sys_bp = st.number_input("Systolic BP", 80, 250, 120)
dia_bp = st.number_input("Diastolic BP", 40, 150, 80)
resp = st.number_input("Respiratory Rate", 5, 60, 16)
spo2 = st.number_input("SpO2 (%)", 70, 100, 97)
temp = st.number_input("Skin Temperature (°C)", 30.0, 42.0, 36.5)

if st.button("Check Risk"):
    input_data = np.array([[hr, sys_bp, dia_bp, resp, spo2, temp]])

    # Scale
    scaled = scaler.transform(input_data)

    # Predict
    score = model.decision_function(scaled)[0]
    label = model.predict(scaled)[0]

    # Interpret
    if label == -1:
        risk = "⚠️ Critical"
    else:
        risk = "✅ Normal"

    st.subheader(f"Risk Level: {risk}")
    st.write(f"Anomaly Score: {score:.3f}")