import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import shap


# Load the trained FNN model and scaler
model = tf.keras.models.load_model("C:/Users/fathi/OneDrive/Desktop/D2 New/Dissertation_HeartDisease/fnn_model.h5")
scaler = joblib.load("C:/Users/fathi/OneDrive/Desktop/D2 New/Dissertation_HeartDisease/model/scaler_framingham.pkl")

# 🚀 **Framingham Risk Score Function**
def framingham_risk_score(age, totChol, sysBP, smoker, diabetes, gender):
    if gender == 1:  # Male
        risk_score = (52.00961 - (20.014077 * np.log(age + 1e-10))  # Avoid log(0)
                      + (0.65304 * np.log(totChol + 1e-10))
                      + (1.90997 * np.log(sysBP + 1e-10))
                      + (0.60138 * smoker)
                      + (0.57367 * diabetes))
        s0 = 0.88936  # Baseline survival probability for men
    else:  # Female
        risk_score = (31.764 - (26.0145 * np.log(age + 1e-10))  # Avoid log(0)
                      + (1.1237 * np.log(totChol + 1e-10))
                      + (2.5536 * np.log(sysBP + 1e-10))
                      + (0.65451 * smoker)
                      + (0.87976 * diabetes))
        s0 = 0.95012  # Baseline survival probability for women

    risk_percentage = 1 - s0 ** np.exp(risk_score)  # Correct formula
    return round(risk_percentage * 100, 2)  # Return percentage


# 🚀 **Set Page Config**
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# 🚀 **Sidebar Navigation**
st.sidebar.title("Navigation 📌")

menu = ["Home",  "🩺 Risk Prediction", "Data Insights","About",]
choice = st.sidebar.selectbox("Choose a Page", menu)

# 🚀 **🏠 Home Page**
if choice == "Home":
    st.title("Heart Disease Prediction System")
    # st.image("heart_banner.jpg", use_column_width=True)  # Add a banner image (optional)
    st.write("""
    Predicts the **10-year risk of CHD (Coronary Heart Disease)** using Machine Learning.
    """)
     # Button to navigate to Risk Prediction Page
    if st.button("Predict Your Risk"):
        st.session_state.page = "🩺 Risk Prediction"
        st.rerun()  # Force rerun to navigate


elif choice == "About":
    st.title("About This Project")
    st.write("""
    This project is designed to predict the **10-year risk of developing heart disease** based on clinical and lifestyle factors.

    ### **Technologies Used**
    - **Machine Learning Models:** FNN, Logistic Regression, Random Forest, Decision Trees and XGBoost
    - **Explainable AI:** SHAP 
    - **Data Preprocessing:** Normalization, Feature Engineering
    - **Frontend:** Streamlit
    
    **👨‍⚕️ Medical Relevance**  
    This tool can help doctors and individuals assess their risk of CHD early, allowing for preventive measures.
    """)

# 🚀 **🩺 Risk Prediction Page**
elif choice == "🩺 Risk Prediction":
    st.title("Predict Your 10-Year CHD Risk")

    # 📌 **Custom CSS for Light Blue Sliders**
    st.markdown(
         """
        <style>
        /* Change slider track color */

        /* Change slider thumb (circle) color */
        div[data-baseweb="slider"] > div > div {
            background: #90e6fc  !important;  /* Adjust for a slightly darker blue 
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.form(key="risk_form"):
        age = st.slider("Age", min_value=20, max_value=90, value=50)
        totChol = st.slider("Total Cholesterol", min_value=100, max_value=400, value=200)
        sysBP = st.slider("Systolic BP", min_value=90, max_value=200, value=120)
        diaBP = st.slider("Diastolic BP", min_value=50, max_value=130, value=80)
        BMI = st.slider("BMI", min_value=15, max_value=50, value=25)
        glucose = st.slider("Glucose Level", min_value=50, max_value=300, value=100)

        currentSmoker = st.checkbox("Current Smoker")
        prevalentHyp = st.checkbox("Has Hypertension")
        diabetes = st.checkbox("Diabetes")

        submit_button = st.form_submit_button("Predict")  # Button inside form

    # Default Missing Features
    sex = 1  # Assume male (1), female (0)
    education = 2  # Default education level
    BPMeds = 0  # BP Medication
    prevalentStroke = 0  # Stroke History
    heartRate = 75  # Default Heart Rate
    cigsPerDay = 0  # Default Cigarettes per day

    # Convert input into numpy array
    user_input = np.array([[age, sex, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
                            diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    # Scale Input
    user_input_scaled = scaler.transform(user_input)

    # Predict CHD Risk & Compute Framingham Risk Score
    if submit_button:  # Runs only when button is clicked
        # 🔍 **Display Debug Info (User Input Values)**
        st.subheader("🛠️ Debug Info: User Input Values")
        st.write("Age:", age)
        st.write("Total Cholesterol:", totChol)
        st.write("Systolic BP:", sysBP)
        st.write("Diastolic BP:", diaBP)
        st.write("BMI:", BMI)
        st.write("Glucose:", glucose)
        st.write("Current Smoker:", currentSmoker)
        st.write("Hypertension:", prevalentHyp)
        st.write("Diabetes:", diabetes)
        prediction = model.predict(user_input_scaled)[0][0] * 100  # Convert to %
        frs_score = framingham_risk_score(age, totChol, sysBP, currentSmoker, diabetes, sex)  # Compute FRS
        
        def classify_risk(score):
            if score < 10:
                return "Low Risk"
            elif 10 <= score <= 20:
                return "Intermediate Risk"
            else:
                return "High Risk"

        ml_risk_category = classify_risk(prediction)
        frs_risk_category = classify_risk(frs_score)


               # Display Results
        st.success(f"ML Model Prediction: {ml_risk_category} ({prediction:.2f}%)")
        st.success(f"Framingham Risk Score: {frs_risk_category} ({frs_score:.2f}%)")
        explainer = shap.Explainer(model, scaler)  # Use the entire dataset
        shap_values = explainer(user_input_scaled)

        # 📌 **SHAP Summary Plot**
        st.subheader("🔍 Feature Importance (SHAP Summary Plot)")
        fig_summary = plt.figure()
        shap.summary_plot(shap_values, user_input_scaled, feature_names=["Age", "Sex", "Education", "Current Smoker", 
                                                                        "Cigs/Day", "BP Meds", "Stroke History", 
                                                                        "Hypertension", "Diabetes", "Total Cholesterol", 
                                                                        "Systolic BP", "Diastolic BP", "BMI", "Heart Rate", "Glucose"])
        st.pyplot(fig_summary)

        # 📌 **SHAP Waterfall Plot**
        st.subheader("📊 Individual Prediction Breakdown (SHAP Waterfall Plot)")
        fig_waterfall = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(fig_waterfall)

# 🚀 **📊 Data Insights Page**
elif choice == "Data Insights":
    st.title("Data Insights")

    # Load Dataset
    heart_data = pd.read_csv("C:/Users/fathi/OneDrive/Desktop/D2 New/Dissertation_HeartDisease/data/framingham.csv")

    # Display Data Summary
    if st.checkbox("Show Data Summary"):
        st.write(heart_data.describe())

    # 🚀 **2. Interactive Histogram: Age Distribution of CHD Patients**
    st.subheader("Age Distribution of CHD Patients")
    fig2 = px.histogram(heart_data, x="age", color="TenYearCHD",
                        title="Age Distribution by CHD Status",
                        labels={"age": "Age", "TenYearCHD": "CHD (1 = Yes, 0 = No)"},
                        nbins=20, barmode="overlay")
    st.plotly_chart(fig2)

    st.subheader("Cholesterol Levels by CHD Status")
    fig3 = px.box(heart_data, x="TenYearCHD", y="totChol", color="TenYearCHD",
                  title="Cholesterol Levels vs. CHD",
                  labels={"TenYearCHD": "CHD Status (0 = No, 1 = Yes)", "totChol": "Total Cholesterol"},
                  points="all")
    st.plotly_chart(fig3)
