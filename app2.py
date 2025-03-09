import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
# import streamlit as st
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
model = tf.keras.models.load_model("cnn_model.h5")
# scaler = joblib.load("scaler_framingham.pkl")
# Extract Expected Features from Scaler
# Load the new scaler that was trained WITHOUT 'education'
scaler = joblib.load("scaler_framingham_updated.pkl")

# Extract expected features
expected_features = scaler.feature_names_in_  # Ensure correct ordering


# Set Streamlit Page Config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
# Apply Custom CSS for Local Background Image
import base64
import streamlit as st

# Function to load and encode image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Convert image to base64
img_base64 = get_base64_of_image("images/img3.jpg")
# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = ["Home", "Risk Prediction", "Data Insights","About"]
choice = st.sidebar.radio("Choose a Page", menu)

if choice == "Home":
    st.session_state.page = "Home"
    st.markdown(
        f"""
        <style>
            .main {{
                background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center fixed;
                background-size: cover;
            }}
            .block-container {{
                background: rgba(255, 255, 255, 0);
                border-radius: 10px;
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }}
            .title {{
                font-size: 55px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }}
            .subtitle {{
                font-size: 35px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.session_state.page = choice  # Update session state for navigation
    st.markdown(
        """
        <style>
            .main {
                background: white !important;
            }
            .block-container {
                background: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

#Home Page
if choice == "Home":
    st.markdown("<div class='title'>CardioRisk AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Heart Disease Prediction System</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:24px; font-weight:bold; color:#333;'>Predicts the <b>10-year risk of CHD (Coronary Heart Disease)</b> using Machine Learning.</p>",
        unsafe_allow_html=True
    )

#  About Page
elif choice == "About":
    st.title("About This Project")
    st.write("""
    ### **CardioRisk AI- Heart Disease Risk Prediction System**
    This application leverages advanced **Machine Learning (ML) algorithms** to estimate an individual's **10-year risk of developing Coronary Heart Disease (CHD)**. By analyzing key health indicators, the system provides a risk assessment that can aid in early detection and preventive care.

    ### **Key Features**
    - **Personalized Risk Assessment:** Predicts the likelihood of developing CHD based on medical and lifestyle factors.
    - **Risk Stratification:** Categorizes individuals into **Low, Medium, or High Risk** using the **Framingham Risk Score**.
    - **Explainable AI (XAI):** Integrates **SHAP** (SHapley Additive Explanations) to provide **insights into how different health factors influence the risk prediction**.
    - **Interactive Visualization:** Displays key **data insights, trends, and feature importance** to enhance interpretability.
    - **AI Powered Insights:** It displays advice to lower risks based on the user's most significant risk.
    
    ### **Technologies Used**
    - **Machine Learning Models:** Convolutional Neural Network (CNN), Feedforward Neural Network (FNN), Logistic Regression, Random Forest, Decision Trees.
    - **Explainability Framework:** SHAP for model interpretability.
    - **Data Preprocessing:** StandardScaler, Feature Engineering for improved accuracy.
    - **User Interface:** Built using **Streamlit** for an intuitive and seamless experience.
    ---
    **Disclaimer:** This application is designed for **educational and informational purposes only** and should not be used as a substitute for professional medical advice. Always consult a healthcare provider for medical guidance.
    
    """)

# Risk Prediction Page
elif choice == "Risk Prediction" or st.session_state.page == "Risk Prediction":
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
    import time
    # redict and Display Results
    if submit_button:
        risk_percentage, risk_category = calculate_framingham_risk(age, gender, totChol, sysBP, currentSmoker, diabetes, BPMeds)
        risk_value = float(risk_percentage.replace('%', '').replace('>30', '30'))
        st.success(f"**Risk Score:** {risk_value} %")
        st.success(f"**Risk Level:** {risk_category}")

        #  Add Progress Bar Before SHAP Explanation
        st.subheader("Explain my risk ")
        progress_bar = st.progress(0)
    
        for percent_complete in range(30, 101, 10):  # Simulated loading effect
            time.sleep(0.2)  # Adjust for a realistic delay
            progress_bar.progress(percent_complete)
                

        # SHAP Explainability
        feature_names = ["male", "age",  "currentSmoker", "cigsPerDay",
                        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
                        "totChol", "sysBP", "diaBP", "BMI",
                        "heartRate", "glucose"]
        explainer = shap.Explainer(model, np.zeros((1, len(expected_features))))  # Matches training data

        shap_values = explainer(user_input_scaled)
        # Extract SHAP Feature Importance (Top 6 Features)
        shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
        top_shap_features = shap_values_df.abs().mean().nlargest(6).index.tolist()
        progress_bar.empty()
        
        # Create two columns for displaying results
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            # SHAP visualization in the left column
            st.subheader("SHAP Explanation")
            fig_summary, ax = plt.subplots(figsize=(3, 3))  # Adjusted for column width
            plt.rcParams.update({'font.size': 6})  # Slightly larger font for readability
            
            # Make sure the plot is compact but readable
            shap.summary_plot(
                shap_values, 
                user_input_df, 
                feature_names=feature_names, 
                show=False,
                 # Show top 7 features for better context
            )
            
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_summary)

            

        with col2:
            # Risk factor and recommendations in the right column
            st.subheader("📌 Top Risk Factor")
            
            # Identify the Most Influential Feature
            shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
            top_risk_factor = shap_values_df.abs().mean().idxmax()
            
            st.write(f"**Most Significant Factor:** `{top_risk_factor}`")
            
            # Display personalized recommendation
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
            
            st.subheader("🩺 Personalized AI Recommendation")
            if top_risk_factor in recommendations:
                st.success(recommendations[top_risk_factor])
            else:
                st.success("Focus on maintaining a heart-healthy lifestyle with regular exercise and a balanced diet.")

            

        with col3:
            st.subheader("LIME Explanation")
            

# Function to Convert Model Predictions into Probability Format for LIME
            def model_predict_proba(X):
                predictions = model.predict(X)  # Get raw predictions
                return np.hstack([1 - predictions, predictions])  # Convert to probability format

            # LIME Uses Full Input Size (Zeros Out Unused Features)
            def masked_input(instance, feature_mask):
                """ Keeps all 14 features but zeroes out the ones not in the SHAP top 6 """
                masked_instance = np.zeros_like(instance)
                for i, feature in enumerate(feature_names):
                    if feature in feature_mask:
                        masked_instance[0][i] = instance[0][i]  # Retain original value
                return masked_instance

            # Create LIME Explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X,  # Use full dataset
                feature_names=feature_names,  # Use all features
                class_names=["No CHD", "CHD Risk"],
                discretize_continuous=False,  # Avoids LIME errors
                mode="classification"
            )

            # Generate LIME Explanation on User Input
            lime_exp = lime_explainer.explain_instance(
                user_input_scaled[0],  # Must match SHAP input
                model_predict_proba,
                num_features=5,  # Show only top SHAP features
                labels=(1,)  # Explain only the CHD Risk class
            )

            # Display LIME Bar Graph
            fig_lime = lime_exp.as_pyplot_figure()
            fig_lime.set_size_inches(4, 3)  # Adjust figure size
            st.pyplot(fig_lime)

        with col4:
            st.header("Tips for a Healthy Heart")
            st.write("💪 **Stay Active:** Regular physical activity helps maintain a healthy weight and lowers the risk of heart disease.")
            st.write("👨‍👩‍👧‍👦 **Know Your Family History:** Understanding your family's health history can help you take preventive measures.")
            st.write("🍎 **Make Heart-Healthy Choices:** Choose a balanced diet rich in fruits, vegetables, and whole grains to support heart health.")
            st.write("💤 **Get Enough Sleep:** Quality sleep is essential for overall health, including heart health. Aim for 7-8 hours of sleep per night.")


# Data Insights Page
elif choice == "Data Insights":
    st.subheader("Heart Disease Data Insights Dashboard")
    st.markdown("### 🩺 Did You Know?")
    st.markdown("- **Regular exercise can reduce the risk of heart disease by up to 50%.**")



    # Create a layout with columns
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)

    box_style = "border: 1px solid lightgrey; padding: 10px; border-radius: 5px;"

    # Age Distribution of CHD Patients
    with col1:
        st.markdown("<div style='" + box_style + "'>", unsafe_allow_html=True)
        fig_age = px.histogram(
            heart_data, x="age", color="TenYearCHD",
            nbins=20, barmode="overlay",
            title="Age Distribution by CHD Status",
            labels={"TenYearCHD": "Heart Disease (0 = No, 1 = Yes)"},
            color_discrete_sequence=["#EF553B", "#636EFA"],  
            opacity=0.7
        )
        st.plotly_chart(fig_age)
        st.markdown("</div>", unsafe_allow_html=True)

    # Cholesterol Levels by CHD Status
    with col2:
        st.markdown("<div style='" + box_style + "'>", unsafe_allow_html=True)
        fig_chol = px.box(
            heart_data, x="TenYearCHD", y="totChol", color="TenYearCHD",
            title="Total Cholesterol Levels by CHD Status",
            points="all", color_discrete_sequence=["#00CC96", "#AB63FA"]
        )
        st.plotly_chart(fig_chol)
        st.markdown("</div>", unsafe_allow_html=True)


    with col3:
        st.markdown("<div style='" + box_style + "'>", unsafe_allow_html=True)
        fig_bp = px.scatter(
            heart_data, x="sysBP", y="diaBP", 
            color="TenYearCHD", 
            title="Blood Pressure Levels: Systolic vs Diastolic",
            labels={"sysBP": "Systolic Blood Pressure (mmHg)", "diaBP": "Diastolic Blood Pressure (mmHg)"},
            color_discrete_sequence=["#636EFA", "#EF553B"], 
            opacity=0.6
        )

        # Update layout
        fig_bp.update_layout(
            xaxis_title="Systolic Blood Pressure (mmHg)",
            yaxis_title="Diastolic Blood Pressure (mmHg)"
        )

        st.plotly_chart(fig_bp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # CVD Risk by Age Group
    with col4:
        st.markdown("<div style='" + box_style + "'>", unsafe_allow_html=True)

        heart_data['Gender'] = heart_data['male'].map({1: 'Male', 0: 'Female'})
        heart_data['Age_Group'] = pd.cut(heart_data['age'], bins=[30, 40, 50, 60, 70, 80],
                                        labels=['30-40', '40-50', '50-60', '60-70', '70-80'])

        # Convert Age_Group to string for proper grouping
        heart_data['Age_Group'] = heart_data['Age_Group'].astype(str)

        # Aggregate data: CHD prevalence by Age Group and Gender, convert to percentage
        age_gender_chd = heart_data.groupby(['Age_Group', 'Gender'])['TenYearCHD'].mean().reset_index()
        age_gender_chd['TenYearCHD'] *= 100  # Convert to percentage
        fig_age_gender = px.bar(
            age_gender_chd, 
            x='Age_Group', 
            y='TenYearCHD', 
            color='Gender',
            barmode='group',
            title="10-Year CHD Risk by Age Group and Gender",
            labels={'TenYearCHD': 'Average CHD Risk (%)', 'Age_Group': 'Age Group'},
            text=age_gender_chd['TenYearCHD'].round(2)  # Display risk values in %
        )
        fig_age_gender.update_traces(textposition='outside')  # Show text outside bars
        fig_age_gender.update_layout(yaxis_title="Average CHD Risk (%)", xaxis_title="Age Group")
        st.plotly_chart(fig_age_gender, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    #  Heart Rate Distribution by CVD Status
    with col5:
        st.markdown("<div style='" + box_style + "'>", unsafe_allow_html=True)
        fig_glucose = px.box(
            heart_data, x="TenYearCHD", y="glucose", color="TenYearCHD",
            title="Glucose Levels by CHD Status",
            labels={"glucose": "Glucose Level (mg/dL)", "TenYearCHD": "CHD Status"},
            color_discrete_sequence=["#FFA15A", "#19D3F3"]
        )

        fig_glucose.update_layout(
            xaxis_title="CHD Status (0 = No, 1 = Yes)",
            yaxis_title="Glucose Level (mg/dL)"
        )
        st.plotly_chart(fig_glucose, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    #  Smoking & Heart Disease Risk
    with col6:
        st.markdown("<div style='" + box_style + "'>", unsafe_allow_html=True)
        fig_smoking = px.histogram(
            heart_data, x="cigsPerDay", color="TenYearCHD",
            title="Cigarettes per Day vs. CHD Risk",
            labels={"cigsPerDay": "Cigarettes per Day", "TenYearCHD": "CHD Status"},
            color_discrete_sequence=["#FF4500", "#1E90FF"], opacity=0.7
        )
        st.plotly_chart(fig_smoking)
        st.markdown("</div>", unsafe_allow_html=True)

    
