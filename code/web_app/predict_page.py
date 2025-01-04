import streamlit as st
import numpy as np
import joblib
# Load the trained model
def load_model():
    return joblib.load("model/fcnn_model.pkl")  # Ensure this file exists
model = load_model()

# Define feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# User input function
def show_predict_page():
    st.title("Heart Disease Prediction")

    st.write("""### Enter patient details to predict the risk of heart disease.""")

    # Input fields
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.slider("Resting ECG (0-2)", 0, 2, 1)
    thalach = st.slider("Max Heart Rate", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    slope = st.slider("Slope (0-2)", 0, 2, 1)
    ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.slider("Thal (1,3,6,7)", 1, 7, 3)

    # Convert user inputs into NumPy array
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(user_input)[0][0]
       # Prediction using Keras model

        if prediction > 0.5:  # If the probability is more than 50%, classify as high risk
            st.error(f"🔴 High Risk of Heart Disease! (Predicted Score: {prediction:.2f})")
        else:
            st.success(f"🟢 Low Risk of Heart Disease. (Predicted Score: {prediction:.2f})")
