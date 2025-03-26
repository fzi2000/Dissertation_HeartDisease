# Cardiovascular Disease Prediction Using Machine Learning

This project uses machine learning models to predict the presence of cardiovascular disease based on patient data. The dataset includes attributes such as age, blood pressure, cholesterol levels, and more. The project incorporates techniques like feature selection, model evaluation, and explainable AI (SHAP) for insights. It focuses on stratifying patients into different risk levels and explaining the risk using SHAP graphs for interpretabiltiy.

---

##  Features
- Exploratory Data Analysis (EDA) to understand feature distributions and correlations.
- Implementation of multiple ML models:
  - Logistic Regression
  - Random Forest
  - Decision Trees
  - Ensemble Model
  - Fully Connected Neural Network (FCNN)
  - Convolutional Neural Networks (1D CNN) 
- Hyperparameter tuning for optimizing model performance.
- Explainable AI using SHAP (SHapley Additive exPlanations) for feature importance and model interpretability.
- K-Fold Cross-Validation for robust model evaluation.
- External Validation

---

## Dataset
- The dataset sourced from Framingham dataset contains clinical attributes such as:
   - age: Age of the patient.
   - male: Gender (1 = male, 0 = female).
   - education: Level of education.
   - currentSmoker: Whether the patient is a current smoker (1 = yes, 0 = no).
   - cigsPerDay: Number of cigarettes smoked per day.
   - BPMeds: Whether the patient is on blood pressure medication (1 = yes, 0 = no).
   - prevalentStroke: History of stroke (1 = yes, 0 = no).
   - prevalentHyp: History of hypertension (1 = yes, 0 = no).
   - diabetes: Whether the patient has diabetes (1 = yes, 0 = no).
   - totChol: Total cholesterol level.
   - sysBP: Systolic blood pressure.
   - diaBP: Diastolic blood pressure.
   - BMI: Body Mass Index.
   - heartRate: Heart rate.
   - glucose: Glucose level.
   - TenYearCHD: Target variable (1 = CHD risk within 10 years, 0 = no CHD risk).

---

## Technologies Used
- **Python** for scripting and analysis.
- **Scikit-Learn** for model training and evaluation.
- **TensorFlow/Keras** for building a Fully Connected Neural Network.
- **SHAP** for explainability.
- **Matplotlib/Seaborn** for data visualization.

---
Due to deep learning experiments being run across different environments, two Git accounts
were used during development, though all contributions originate from the same author. 


##  Installation
1. Download the zipped file
2. Extract the folder
3. Install the required dependencies
4. To view the app, go to the terminal in PS C:\Users\fathi\Downloads\Dissertation_HeartDisease-main\Dissertation_HeartDisease-main (file location)
5. Type  "streamlit run app2.py" and click enter to run

   OR 

 Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction

   
