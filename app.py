import streamlit as st
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="Rainfall Prediction", layout="centered")
st.title("🌧️ Rainfall Prediction App")
st.write("Predict rainfall based on daily weather parameters using the best Logistic Regression model.")

# Feature names in correct order
feature_names = [
    'pressure', 
    'temparature',
    'dewpoint',
    'humidity',
    'cloud',
    'sunshine',
    'winddirection',
    'windspeed'
]

# Input fields
st.markdown("### Enter weather conditions")
user_inputs = []
cols = st.columns(4)
for i, feature in enumerate(feature_names):
    val = cols[i % 4].number_input(f"{feature}", value=0.0, format="%.3f")
    user_inputs.append(val)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("rainfall_logistic_model.pkl")
    return model

model = load_model()

# Prediction
if st.button("Predict Rainfall"):
    X_input = np.array(user_inputs).reshape(1, -1)
    probability = model.predict_proba(X_input)[0, 1]
    prediction = model.predict(X_input)[0]

    st.subheader("Prediction Results")
    st.write(f"**Predicted Probability of Rain:** {probability:.3f}")

    if prediction == 1:
        st.success("🌧️ Rain expected!")
    else:
        st.info("☀️ No rain expected.")
