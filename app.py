import streamlit as st
import pandas as pd
import joblib

# Load trained files
model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the medical details below to predict heart disease risk.")

# ---------------- INPUTS ---------------- #

age = st.slider("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "NSY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.number_input(
    "Max Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=6.0,
    step=0.1
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# ---------------- PREDICT BUTTON ---------------- #
if st.button("Predict"):

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    # Convert to dataframe
    df = pd.DataFrame([data])

    # One hot encode
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Scale numeric columns
    numerical_cols = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Prediction
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ No Heart Disease Detected")