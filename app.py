
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model()

# Title
st.title("❤️ Heart Disease Prediction System")
st.markdown("### AI-Powered Heart Disease Risk Assessment")
st.markdown("---")

# Sidebar - Patient Information
st.sidebar.header("📋 Patient Information")

# Input fields
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    options=["ASY (Asymptomatic)", "ATA (Atypical Angina)", 
             "NAP (Non-Anginal Pain)", "TA (Typical Angina)"]
)
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 
                                     min_value=80, max_value=200, value=120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dl)", 
                                      min_value=0, max_value=600, value=200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                                  options=["No", "Yes"])
resting_ecg = st.sidebar.selectbox(
    "Resting ECG",
    options=["Normal", "ST (ST-T wave abnormality)", "LVH (Left Ventricular Hypertrophy)"]
)
max_hr = st.sidebar.number_input("Maximum Heart Rate", 
                                 min_value=60, max_value=220, value=150)
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", 
                                       options=["No", "Yes"])
oldpeak = st.sidebar.number_input("Oldpeak (ST Depression)", 
                                  min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", 
                                options=["Up", "Flat", "Down"])

# Predict button
predict_button = st.sidebar.button("🔍 Predict", use_container_width=True)

# Main area - Information
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 About This System")
    st.write("""
    This AI system predicts heart disease risk using machine learning.
    
    **Model Performance:**
    - Accuracy: 90.76%
    - Recall: 94.12% (catches 94% of diseases!)
    - Algorithm: K-Nearest Neighbors
    
    **Features Used:**
    - Patient demographics (Age, Sex)
    - Clinical measurements (BP, Cholesterol, Heart Rate)
    - Symptoms (Chest Pain, Exercise Angina)
    - ECG results (Resting ECG, ST Slope, Oldpeak)
    """)

with col2:
    st.markdown("### ⚠️ Important Notes")
    st.warning("""
    **Medical Disclaimer:**
    - This is a screening tool, NOT a diagnosis
    - Always consult a healthcare professional
    - Emergency symptoms require immediate medical attention
    
    **Data Privacy:**
    - No data is stored or transmitted
    - All predictions are local
    """)

st.markdown("---")

# Prediction logic
if predict_button:
    # Convert categorical to numerical
    sex_encoded = 1 if sex == "Male" else 0
    fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
    exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
    
    # Chest Pain Type encoding
    chest_pain_map = {
        "ASY (Asymptomatic)": [0, 0, 0],
        "ATA (Atypical Angina)": [1, 0, 0],
        "NAP (Non-Anginal Pain)": [0, 1, 0],
        "TA (Typical Angina)": [0, 0, 1]
    }
    chest_encoded = chest_pain_map[chest_pain]
    
    # Resting ECG encoding
    ecg_map = {
        "Normal": [1, 0],
        "ST (ST-T wave abnormality)": [0, 1],
        "LVH (Left Ventricular Hypertrophy)": [0, 0]
    }
    ecg_encoded = ecg_map[resting_ecg]
    
    # ST Slope encoding
    slope_map = {
        "Up": [0, 1],
        "Flat": [1, 0],
        "Down": [0, 0]
    }
    slope_encoded = slope_map[st_slope]
    
    # Missing indicators
    cholesterol_missing = 1 if cholesterol == 0 else 0
    resting_bp_missing = 1 if resting_bp == 0 else 0
    
    # Handle zero values
    if cholesterol == 0:
        cholesterol = 237.0
    if resting_bp == 0:
        resting_bp = 130.0
    
    # Create feature array (17 features)
    features = [
        age, sex_encoded, resting_bp, cholesterol, fasting_bs_encoded,
        max_hr, exercise_angina_encoded, oldpeak,
        cholesterol_missing, resting_bp_missing,
        chest_encoded[0], chest_encoded[1], chest_encoded[2],
        ecg_encoded[0], ecg_encoded[1],
        slope_encoded[0], slope_encoded[1]
    ]
    
    # Convert to array and reshape
    input_data = np.array(features).reshape(1, -1)
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("## 🎯 Prediction Results")
    
    if prediction == 1:
        st.error("### ⚠️ HIGH RISK - Heart Disease Detected")
        risk_percentage = prediction_proba[1] * 100
        st.metric("Risk Level", f"{risk_percentage:.1f}%", delta="High Risk")
        
        st.markdown("""
        ### 🏥 Recommended Actions:
        1. **Consult a cardiologist immediately**
        2. Schedule comprehensive cardiac tests
        3. Review lifestyle factors
        4. Monitor vitals regularly
        5. Follow prescribed medications
        """)
    else:
        st.success("### ✅ LOW RISK - No Heart Disease Detected")
        risk_percentage = prediction_proba[0] * 100
        st.metric("Risk Level", f"{100-risk_percentage:.1f}%", delta="Low Risk")
        
        st.markdown("""
        ### 💚 Preventive Measures:
        1. Maintain healthy lifestyle
        2. Regular exercise (150 min/week)
        3. Balanced diet
        4. Annual checkups
        5. Manage stress
        """)
    
    # Input summary
    st.markdown("---")
    st.markdown("### 📝 Input Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Age:** {age} years")
        st.write(f"**Sex:** {sex}")
        st.write(f"**Chest Pain:** {chest_pain.split('(')[0].strip()}")
        st.write(f"**Resting BP:** {resting_bp} mm Hg")
    
    with col2:
        st.write(f"**Cholesterol:** {cholesterol} mg/dl")
        st.write(f"**Fasting BS:** {fasting_bs}")
        st.write(f"**Max Heart Rate:** {max_hr} bpm")
        st.write(f"**Exercise Angina:** {exercise_angina}")
    
    with col3:
        st.write(f"**Resting ECG:** {resting_ecg.split('(')[0].strip()}")
        st.write(f"**Oldpeak:** {oldpeak}")
        st.write(f"**ST Slope:** {st_slope}")
    
    # Confidence
    st.markdown("---")
    st.markdown("### 📊 Model Confidence")
    
    conf_col1, conf_col2 = st.columns(2)
    
    with conf_col1:
        st.metric("Healthy Probability", f"{prediction_proba[0]*100:.2f}%")
    with conf_col2:
        st.metric("Disease Probability", f"{prediction_proba[1]*100:.2f}%")
    
    st.progress(prediction_proba[0], text="Healthy")
    st.progress(prediction_proba[1], text="Disease")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | KNN Model (90.76% Accuracy)</p>
    <p><em>Educational purposes only. Consult healthcare professionals.</em></p>
</div>
""", unsafe_allow_html=True)
