import streamlit as st
import pandas as pd
import numpy as np
import joblib
import csv
import os

# --- DATA LOGGING FUNCTION ---
def save_to_csv(patient_dict, calculated_risk):
    file_name = "patient_records.csv" #  LOCAL SPREADSHEET PATH

    record_data = patient_dict.copy()
    record_data['Calculated_Risk_Percentage'] = round(calculated_risk, 2)
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=record_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record_data)

# --- APP CONFIGURATION ---
st.set_page_config(page_title="CardioRisk AI", page_icon="❤️", layout="centered")
st.title("❤️ CardioRisk AI: 10-Year CVD Clinical Predictor")

@st.cache_resource
def load_medical_model():
    return joblib.load("cvd_risk_scoring_model.pkl") #  CORRECT LOCAL PATH


try:
    model = load_medical_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.header("📋 Patient Profile & Metrics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=45)
    male = st.selectbox("Biological Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.slider("Education Level (Category 1-4)", min_value=1, max_value=4, value=2)
    current_smoker = st.selectbox("Current Smoker?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cigs_per_day = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0 if current_smoker==0 else 10)

with col2:
    bp_meds = st.selectbox("On Blood Pressure Medication?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    prev_stroke = st.selectbox("History of Stroke?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    prev_hyp = st.selectbox("History of Hypertension?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diagnosed with Diabetes?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.subheader("🩸 Vital Signs & Biomarkers")
col3, col4, col5 = st.columns(3)

with col3:
    tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=220)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)

with col4:
    sys_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
    knows_glucose = st.checkbox("Have Glucose data?", value=True)
    glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=500, value=85) if knows_glucose else np.nan

with col5:
    dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.4, step=0.1)

patient_data = {
    'male': male, 'age': age, 'education': education, 'currentSmoker': current_smoker,
    'cigsPerDay': cigs_per_day, 'BPMeds': bp_meds, 'prevalentStroke': prev_stroke,
    'prevalentHyp': prev_hyp, 'diabetes': diabetes, 'totChol': tot_chol,
    'sysBP': sys_bp, 'diaBP': dia_bp, 'BMI': bmi, 'heartRate': heart_rate, 'glucose': glucose
}

if st.button("📊 Calculate Patient Risk Score", type="primary"):
    patient_df = pd.DataFrame([patient_data])
    risk_proba = model.predict_proba(patient_df)
    risk_percentage = float(risk_proba[0][1] * 100)
    
    save_to_csv(patient_data, risk_percentage)
    
    st.markdown("---")
    st.subheader("🩺 Clinical Risk Assessment")
    
    if risk_percentage < 10:
        st.success(f"### Risk: {risk_percentage:.1f}% — LOW RISK")
    elif risk_percentage < 20:
        st.warning(f"### Risk: {risk_percentage:.1f}% — MODERATE RISK")
    else:
        st.error(f"### Risk: {risk_percentage:.1f}% — HIGH RISK")
