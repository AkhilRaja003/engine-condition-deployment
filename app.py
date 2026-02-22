
import streamlit as st
import joblib
import numpy as np

st.title("Engine Condition Predictor")

@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

st.write("Enter values to predict engine condition")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")

if st.button("Predict"):
    try:
        pred = model.predict([[f1, f2, f3]])
        st.success(f"Prediction: {pred[0]}")
    except Exception as e:
        st.error("Prediction failed 👇")
        st.error(str(e))
