import numpy as np
import joblib

model_path = "C:/Users/fathi/OneDrive/Desktop/D2/Dissertation_HeartDisease/model/fcnn_model.pkl"

# Load the trained model
# model_path = "model/fcnn_model.pkl"  # Ensure this path is correct
try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"❌ Error: Model file '{model_path}' not found. Check the path and try again.")
    exit()

# Define feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Ask for user inputs
user_input = []
print("\n🔹 Enter Patient Details for Heart Disease Prediction 🔹")
for feature in feature_names:
    value = float(input(f"Enter {feature}: "))
    user_input.append(value)

# Convert inputs into NumPy array
user_input = np.array([user_input])

# Run prediction
print("\n🔄 Running Prediction...")

try:
    prediction = model.predict(user_input)[0][0]  # Ensure the model output format is correct
except AttributeError as e:
    print(f"❌ Model error: {e}")
    print("🔹 This error usually occurs if the model is a Keras `Sequential` model. Try using `.predict()` instead.")
    exit()

# Display results
print("\n✅ Prediction Complete!")
if prediction > 0.5:
    print(f"🔴 High Risk of Heart Disease! (Predicted Score: {prediction:.2f})")
else:
    print(f"🟢 Low Risk of Heart Disease. (Predicted Score: {prediction:.2f})")
