import numpy as np
import joblib
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns
# import ace_tools as tools  # For displaying results

# Disable TensorFlow OneDNN optimization warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("✅ Script started...")

# Load model and scaler
model_path = "C:/Users/fathi/OneDrive/Desktop/D2 New/Dissertation_HeartDisease/model/best_model.pkl"
scaler_path = "C:/Users/fathi/OneDrive/Desktop/D2 New/Dissertation_HeartDisease/model/scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"✅ Model Type: {type(model)}")
    print(f"✅ Scaler Type: {type(scaler)}")
    print("✅ Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# Dataset (input values and actual labels)
data = [
    [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0.0, 6.0, 0],
    [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3.0, 3.0, 2],
    [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2.0, 7.0, 1],
    [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0.0, 3.0, 0],
    [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0.0, 3.0, 0],
    [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0.0, 3.0, 0],
    [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2.0, 3.0, 3],
    [57, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, 0.0, 3.0, 0],
]

# Convert to NumPy array and separate features from labels
data_np = np.array(data)
X_test = data_np[:, :-1]  # Input features
y_test = data_np[:, -1]   # Actual labels

# Apply feature scaling
X_test_scaled = scaler.transform(X_test)

# Make predictions
predictions = model.predict(X_test_scaled).flatten()

# **Step 1: Convert Model Output into 3 Risk Levels**
def stratify_risk(prob):
    if prob >= 0.60:
        return "High Risk 🔴"
    elif prob >= 0.5:
        return "Medium Risk 🟡"
    else:
        return "Low Risk 🟢"

predicted_risks = [stratify_risk(p) for p in predictions]

# **Step 2: Map Dataset Labels (0-3) to 3 Risk Levels**
def map_actual_risk(label):
    if label == 0:
        return "Low Risk 🟢"
    elif label == 1:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"

actual_risks = [map_actual_risk(y) for y in y_test]

# **Step 3: Create a DataFrame for Analysis**
comparison_df = pd.DataFrame({
    'Actual Num': y_test,  # Original dataset labels (0-3)
    'Expected Risk': actual_risks,  # Converted expected risk
    'Predicted Score': predictions,  # Model output probability
    'Predicted Risk': predicted_risks  # Risk level from model
})

# **Display the DataFrame in user-friendly format**
print("\n🔍 Prediction Results:\n")
print(comparison_df)
# **Step 4: Compute Accuracy Metrics**
risk_mapping = {"Low Risk 🟢": 0, "Medium Risk 🟡": 1, "High Risk 🔴": 2}
comparison_df['Expected Risk Num'] = comparison_df['Expected Risk'].map(risk_mapping)
comparison_df['Predicted Risk Num'] = comparison_df['Predicted Risk'].map(risk_mapping)

# Generate classification report
print("\n🔍 Classification Report:")
print(classification_report(comparison_df['Expected Risk Num'], comparison_df['Predicted Risk Num']))

# **Step 5: Confusion Matrix for Error Analysis**
# cm = confusion_matrix(comparison_df['Expected Risk Num'], comparison_df['Predicted Risk Num'])
# plt.figure(figsize=(6,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=risk_mapping.keys(), yticklabels=risk_mapping.keys())
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix for Risk Levels")
# plt.show()

# # **Step 6: Investigate Wrong Predictions**
# wrong_preds = comparison_df[comparison_df["Expected Risk"] != comparison_df["Predicted Risk"]]
# tools.display_dataframe_to_user(name="Incorrect Predictions", dataframe=wrong_preds)

# print("✅ All predictions completed successfully!")
