import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# import streamlit as st
# from streamlit_chat import message
# from transformers import pipeline

# # Load Chatbot Model (small model for performance)
# chatbot_model = pipeline("text-generation", model="distilgpt2")

# # Initialize Chat History
# if "messages" not in st.session_state:
#     st.session_state["messages"] = [
#         {"role": "assistant", "content": "Hello! I'm your AI assistant. Ask me anything about heart health! 😊"}
#     ]


import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
heart_data = pd.read_csv('data/framingham.csv')

# Drop 'education' column before training the scaler
X = heart_data.drop(columns=["TenYearCHD", "education"])  # Exclude target & 'education'

# Fit a new scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the updated scaler
joblib.dump(scaler, "scaler_framingham_updated.pkl")

# # Scale the data again
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
model = tf.keras.models.load_model("fnn_model.h5")
# scaler = joblib.load("scaler_framingham.pkl")
# Extract Expected Features from Scaler
# Load the new scaler that was trained WITHOUT 'education'
scaler = joblib.load("scaler_framingham_updated.pkl")

# Extract expected features
expected_features = scaler.feature_names_in_  # Ensure correct ordering


# Set Streamlit Page Config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation 📌")
menu = ["Home", "Risk Prediction", "Data Insights", "ChatBot","About"]
choice = st.sidebar.radio("Choose a Page", menu)

#Home Page
if choice == "Home":
    st.title("Heart Disease Prediction System")
    st.write("Predicts the **10-year risk of CHD (Coronary Heart Disease)** using Machine Learning.")

    if st.button("Predict Your Risk"):
        st.session_state.page = "Risk Prediction"
        st.rerun()

#  About Page
elif choice == "About":
    st.title("About This Project")
    st.write("""
    ### **Heart Disease Risk Prediction System**
    This application leverages advanced **Machine Learning (ML) algorithms** to estimate an individual's **10-year risk of developing Coronary Heart Disease (CHD)**. By analyzing key health indicators, the system provides a risk assessment that can aid in early detection and preventive care.

    ### **Key Features**
    - **Personalized Risk Assessment:** Predicts the likelihood of developing CHD based on medical and lifestyle factors.
    - **Risk Stratification:** Categorizes individuals into **Low, Medium, or High Risk** using the **Framingham Risk Score**.
    - **Explainable AI (XAI):** Integrates **SHAP** (SHapley Additive Explanations) to provide **insights into how different health factors influence the risk prediction**.
    - **Interactive Visualization:** Displays key **data insights, trends, and feature importance** to enhance interpretability.
    
    ### **Technologies Used**
    - **Machine Learning Models:** Feedforward Neural Network (FNN), Logistic Regression, Random Forest, XGBoost.
    - **Explainability Framework:** SHAP for model interpretability.
    - **Data Preprocessing:** StandardScaler, Feature Engineering for improved accuracy.
    - **User Interface:** Built using **Streamlit** for an intuitive and seamless experience.
    ---
    **Disclaimer:** This application is designed for **educational and informational purposes only** and should not be used as a substitute for professional medical advice. Always consult a healthcare provider for medical guidance.
    
    """)

# Risk Prediction Page
elif choice == "Risk Prediction":
    st.title("Predict Your 10-Year CHD Risk")

    with st.form(key="risk_form"):
        age = st.slider("Age", 20, 90, 50)
        gender = st.radio("Select Gender", ["Male", "Female"])
        totChol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
        sysBP = st.slider("Systolic BP (mmHg)", 90, 200, 120)
        diaBP = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
        BMI = st.slider("BMI", 15, 50, 25)
        glucose = st.slider("Glucose Level (mg/dL)", 50, 300, 100)
        cigsPerDay = st.slider("Cigarettes per day", 0, 70, 0)
        currentSmoker = st.checkbox("Current Smoker")
        prevalentHyp = st.checkbox("Has Hypertension")
        prevalentStroke = st.checkbox("Has History of Stroke")
        diabetes = st.checkbox("Diabetes")
        BPMeds = st.checkbox("BP Medication")

        submit_button = st.form_submit_button("Predict")

    # Ensure Correct Feature Order
    sex = 1 if gender == "Male" else 0
    heartRate = 75  

    # Ensure feature order is correct (excluding 'education')
    user_input_df = pd.DataFrame([[sex, age, currentSmoker, cigsPerDay,
                                BPMeds, prevalentStroke, prevalentHyp, diabetes,
                                totChol, sysBP, diaBP, BMI, heartRate, glucose]],
                                columns=expected_features)

    # Scale input data
    user_input_scaled = scaler.transform(user_input_df)  # This should now work!

    # Create DataFrame with Correct Feature Names & Order
# # Create user input DataFrame WITHOUT 'education'
#     user_input_df = pd.DataFrame([[sex, age, currentSmoker, cigsPerDay,
#                                BPMeds, prevalentStroke, prevalentHyp, diabetes,
#                                totChol, sysBP, diaBP, BMI, heartRate, glucose]],
#                              columns=[col for col in expected_features if col != "education"])


#     # Scale the input
#     user_input_scaled = scaler.transform(user_input_df)

    # Framingham Risk Score Calculation
    def calculate_framingham_risk(age, sex, totChol, sysBP, currentSmoker, diabetes, BPMeds):
        points = 0

        # Assign Points for Age
        age_points = 0
        if sex == "Male":
            if 30 <= age < 35:
                age_points = 0
            elif 35 <= age < 40:
                age_points = 2
            elif 40 <= age < 45:
                age_points = 5
            elif 45 <= age < 50:
                age_points = 6
            elif 50 <= age < 55:
                age_points = 8
            elif 55 <= age < 60:
                age_points = 10
            elif 60 <= age < 65:
                age_points = 11
            elif 65 <= age < 70:
                age_points = 12
            elif 70 <= age < 75:
                age_points = 14
            else:
                age_points = 15  # 75+
        else:  # Female
            if 30 <= age < 35:
                age_points = 0
            elif 35 <= age < 40:
                age_points = 2
            elif 40 <= age < 45:
                age_points = 4
            elif 45 <= age < 50:
                age_points = 5
            elif 50 <= age < 55:
                age_points = 7
            elif 55 <= age < 60:
                age_points = 8
            elif 60 <= age < 65:
                age_points = 9
            elif 65 <= age < 70:
                age_points = 10
            elif 70 <= age < 75:
                age_points = 11
            else:
                age_points = 12  # 75+
        
        points += age_points

        # Assign Points for Total Cholesterol
        chol_points = 0
        if totChol < 160:
            chol_points = 0
        elif 160 <= totChol < 200:
            chol_points = 1
        elif 200 <= totChol < 240:
            chol_points = 2
        elif 240 <= totChol < 280:
            chol_points = 3
        else:
            chol_points = 4
        
        points += chol_points

        # Assign Points for Systolic BP
        bp_points = 0
        if not BPMeds:
            if sysBP < 120:
                bp_points = 0
            elif 120 <= sysBP < 130:
                bp_points = 1
            elif 130 <= sysBP < 140:
                bp_points = 2
            elif 140 <= sysBP < 160:
                bp_points = 3
            else:
                bp_points = 4
        else:
            if sysBP < 120:
                bp_points = 0
            elif 120 <= sysBP < 130:
                bp_points = 3
            elif 130 <= sysBP < 140:
                bp_points = 4
            elif 140 <= sysBP < 160:
                bp_points = 5
            else:
                bp_points = 6
        
        points += bp_points

        # Assign Points for Smoking
        smoker_points = 4 if currentSmoker else 0
        points += smoker_points

        # Assign Points for Diabetes
        diabetes_points = 3 if diabetes else 0
        points += diabetes_points

        # Official Framingham Risk Mapping (Men & Women)
        risk_mapping_men = {
            -3: "<1", -2: "1.1%", -1: "1.4%", 0: "1.6%", 1: "1.9%", 2: "2.3%", 3: "2.8%",
            4: "3.3%", 5: "3.9%", 6: "4.7%", 7: "5.6%", 8: "6.7%", 9: "7.9%", 10: "9.4%",
            11: "11.2%", 12: "13.3%", 13: "15.6%", 14: "18.4%", 15: "21.6%", 16: "25.3%",
            17: "29.4%", 18: ">30%", 19: ">30%", 20: ">30%", 21: ">30%"
        }

        risk_mapping_women = {
            -3: "<1", -2: "<1", -1: "1.0%", 0: "1.2%", 1: "1.5%", 2: "1.7%", 3: "2.0%",
            4: "2.4%", 5: "2.8%", 6: "3.3%", 7: "3.9%", 8: "4.5%", 9: "5.3%", 10: "6.3%",
            11: "7.3%", 12: "8.6%", 13: "10.0%", 14: "11.7%", 15: "13.7%", 16: "15.9%",
            17: "18.5%", 18: "21.5%", 19: "24.8%", 20: "27.5%", 21: ">30%"
        }

        #  Assign Risk Based on Gender
        risk_percentage = risk_mapping_men.get(points, ">30%") if sex == "Male" else risk_mapping_women.get(points, ">30%")

        # Convert Risk to Numeric for Categorization
        risk_value = float(risk_percentage.replace('%', '').replace('>30', '30'))

        # Risk Stratification
        if risk_value < 10:
            risk_category = "🟢 Low Risk (Less than 10%)"
        elif 10 <= risk_value < 20:
            risk_category = "🟡 Medium Risk (10% - 20%)"
        else:
            risk_category = "🔴 High Risk (Above 20%)"

        return risk_percentage, risk_category

    # redict and Display Results
    if submit_button:
        risk_percentage, risk_category = calculate_framingham_risk(age, gender, totChol, sysBP, currentSmoker, diabetes, BPMeds)
        risk_value = float(risk_percentage.replace('%', '').replace('>30', '30'))
        st.success(f"**Risk Score:** {risk_value} %")
        st.success(f"**Risk Level:** {risk_category}")

        # SHAP Explainability
        feature_names = ["male", "age",  "currentSmoker", "cigsPerDay",
                        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
                        "totChol", "sysBP", "diaBP", "BMI",
                        "heartRate", "glucose"]

        explainer = shap.Explainer(model, np.zeros((1, len(expected_features))))  # Matches training data

        shap_values = explainer(user_input_scaled)

        # Generate SHAP Summary Plot with Correct Labels
        st.subheader("Explain my risk (SHAP Summary Plot)")
        fig_summary = plt.figure()
        shap.summary_plot(shap_values, user_input_df, feature_names=feature_names)  
        st.pyplot(fig_summary)

        # 🚀 Identify the Most Influential Feature
        shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
        top_risk_factor = shap_values_df.abs().mean().idxmax()

        st.subheader("📌 Top Risk Factor Influencing Your Prediction")
        st.write(f"**Most Significant Factor:** `{top_risk_factor}`")

        # 🚀 AI-Powered Treatment Recommendations
        recommendations = {
            "cigsPerDay": "🚭 **Reduce smoking**: Consider a smoking cessation program or nicotine replacement therapy.",
            "currentSmoker": "🚭 **Quit smoking**: Seek support from a healthcare provider to stop smoking permanently.",
            "totChol": "🥗 **Adopt a heart-healthy diet**: Reduce saturated fats, eat more fruits and vegetables.",
            "sysBP": "💊 **Monitor blood pressure**: Reduce salt intake, exercise regularly, and consider medication if necessary.",
            "diaBP": "💊 **Control diastolic pressure**: Lower stress, avoid caffeine, and maintain a balanced diet.",
            "BMI": "⚖️ **Maintain a healthy weight**: Follow a balanced diet and engage in regular physical activity.",
            "glucose": "🍬 **Monitor blood sugar levels**: Reduce sugar intake, exercise, and consider medication if diabetic.",
            "diabetes": "🍏 **Manage diabetes**: Control carbohydrate intake and consult a doctor for diabetes management.",
            "prevalentHyp": "💊 **Control hypertension**: Follow a low-sodium diet and engage in moderate exercise.",
            "prevalentStroke": "⚠️ **Stroke prevention**: Avoid smoking, control cholesterol, and maintain normal blood pressure.",
            "BPMeds": "💊 **Consult your doctor about blood pressure medications**: Ensure correct dosage and adherence.",
            "age": "🏥 **Regular health check-ups**: Monitor cardiovascular health with routine medical exams.",
            "male": "⚠️ **Be aware of male-specific heart risks**: Men have a higher risk of CHD; focus on preventive care.",
            "education": "📖 **Stay informed about heart health**: Knowledge helps in making proactive health choices.",
            "heartRate": "💓 **Maintain a normal heart rate**: Regular exercise and stress management are beneficial."
        }

        # 🚀 Display the AI-Powered Recommendation
        if top_risk_factor in recommendations:
            st.subheader("🩺 Personalized AI Recommendation")
            st.success(recommendations[top_risk_factor])



# Data Insights Page
elif choice == "Data Insights":
    st.title("Data Insights")

    heart_data = pd.read_csv("data/framingham.csv")

    if st.checkbox("Show Data Summary"):
        st.write(heart_data.describe())

# 🎨 **Age Distribution of CHD Patients**
    st.subheader("📊 Age Distribution of CHD Patients")
    fig_age = px.histogram(
        heart_data, x="age", color="TenYearCHD",
        nbins=20, barmode="overlay",
        title="Age Distribution by CHD Status",
        labels={"TenYearCHD": "Heart Disease (0 = No, 1 = Yes)"},
        color_discrete_sequence=["#EF553B", "#636EFA"],  # Red & Blue
        opacity=0.7
    )
    fig_age.update_layout(
        template="plotly_dark",
        xaxis=dict(title="Age"),
        yaxis=dict(title="Count"),
        legend=dict(title="CHD Status")
    )
    st.plotly_chart(fig_age)

    # 🎨 **Cholesterol Levels by CHD Status**
    st.subheader("🩸 Cholesterol Levels by CHD Status")
    fig_chol = px.box(
        heart_data, x="TenYearCHD", y="totChol", color="TenYearCHD",
        title="Total Cholesterol Levels by CHD Status",
        points="all", color_discrete_sequence=["#00CC96", "#AB63FA"]
    )
    fig_chol.update_layout(
        template="presentation",
        xaxis=dict(title="CHD Status"),
        yaxis=dict(title="Total Cholesterol"),
    )
    st.plotly_chart(fig_chol)

    # 🎨 **Systolic Blood Pressure Distribution by CVD Status**
    st.subheader("💙 Blood Pressure vs. Heart Disease")
    fig_bp = px.violin(
        heart_data, x="TenYearCHD", y="sysBP", color="TenYearCHD", box=True,
        labels={"TenYearCHD": "CVD Status", "sysBP": "Systolic BP"},
        title="Systolic Blood Pressure Distribution",
        color_discrete_sequence=["#FFA07A", "#4682B4"]
    )
    fig_bp.update_layout(
        template="plotly_white",
        yaxis=dict(title="Systolic Blood Pressure"),
    )
    st.plotly_chart(fig_bp)

    # 🎨 **BMI vs. Heart Disease Risk**
    st.subheader("⚖️ BMI vs. CVD Risk")
    fig_bmi = px.scatter(
        heart_data, x="BMI", y="TenYearCHD", color="TenYearCHD",
        title="BMI vs. Heart Disease Risk",
        color_discrete_sequence=["#FFD700", "#FF4500"],
        size_max=10, opacity=0.8
    )
    fig_bmi.update_layout(
        template="plotly_dark",
        xaxis=dict(title="BMI"),
        yaxis=dict(title="CVD Status (0 = No, 1 = Yes)"),
    )
    st.plotly_chart(fig_bmi)

    # 🎨 **Heart Rate Distribution by CVD Status**
    st.subheader("💓 Heart Rate & CVD Status")
    fig_hr = px.box(
        heart_data, x="TenYearCHD", y="heartRate", color="TenYearCHD",
        title="Heart Rate Distribution",
        color_discrete_sequence=["#D62728", "#1F77B4"]
    )
    fig_hr.update_layout(
        template="ggplot2",
        xaxis=dict(title="CHD Status"),
        yaxis=dict(title="Heart Rate"),
    )
    st.plotly_chart(fig_hr)



# Import the dataset
heart_data = pd.read_csv('data/framingham.csv')
# Print the first 10 lines of the dataset
print(heart_data.head(10)) 
# Display basic information about dataset
heart_data.info()
# Checking the data shape
heart_data.shape
# Check for Missing Values
missing_values=heart_data.isnull().sum()
print(pd.DataFrame({'Missing Values': missing_values}))
# Remove rows with missing values
heart_data = heart_data.dropna()
# Verify if all missing values are removed
print("Missing Values After Removal:\n", heart_data.isnull().sum())
# Check for duplicate records
duplicates = heart_data.duplicated().sum()
print(f"\nDuplicate Records Found: {duplicates}")
# Remove duplicates if any exist
if duplicates > 0:
    heart_data = heart_data.drop_duplicates()
    print("Duplicates Removed")