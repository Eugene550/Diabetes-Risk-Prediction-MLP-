# ============================================================
# Diabetes Risk Prediction — INTERACTIVE PREDICTION SCRIPT
# BAXI 3133 | Neural Network | Group Mini Project
# ------------------------------------------------------------
# This script LOADS the already-trained ANN model and lets the
# user enter patient details to get a diabetes risk prediction.
# NO retraining required — gives identical results every time.
#
# Required files (must be in the same folder as this script):
#   - diabetes_ann_model.keras   (trained ANN model)
#   - diabetes_scaler.pkl        (StandardScaler)
#   - diabetes_le_gender.pkl     (gender encoder)
#   - diabetes_le_smoking.pkl    (smoking encoder)
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import warnings
warnings.filterwarnings('ignore')  # Suppress harmless sklearn warnings

import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ============================================================
# LOAD THE TRAINED MODEL AND PREPROCESSING OBJECTS
# ============================================================
print("=" * 55)
print("  DIABETES RISK PREDICTION — Loading Trained Model")
print("=" * 55)

model      = load_model('diabetes_ann_model.keras')
scaler     = joblib.load('diabetes_scaler.pkl')
le_gender  = joblib.load('diabetes_le_gender.pkl')
le_smoking = joblib.load('diabetes_le_smoking.pkl')

# Valid category options (taken from the encoders)
VALID_GENDER  = list(le_gender.classes_)    # ['Female', 'Male']
VALID_SMOKING = list(le_smoking.classes_)   # ['No Info','current','ever','former','never','not current']

print("Model and preprocessing objects loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_diabetes(gender, age, hypertension, heart_disease,
                     smoking_history, bmi, HbA1c_level, blood_glucose_level):
    """Predict diabetes risk for a new patient using the trained ANN."""
    gender_enc  = le_gender.transform([gender])[0]
    smoking_enc = le_smoking.transform([smoking_history])[0]

    features = np.array([[gender_enc, age, hypertension, heart_disease,
                          smoking_enc, bmi, HbA1c_level, blood_glucose_level]])
    features_scaled = scaler.transform(features)

    proba = float(model.predict(features_scaled, verbose=0)[0][0])

    # Cap at 99% — no model should ever claim 100% certainty
    proba_display = min(round(proba * 100, 2), 99.0)

    return {
        'prediction'  : 'Diabetic' if proba > 0.5 else 'Not Diabetic',
        'probability' : proba_display,
        'risk_level'  : 'HIGH'   if proba > 0.70 else
                        'MEDIUM' if proba > 0.40 else 'LOW'
    }


# ============================================================
# INPUT VALIDATION HELPERS
# ============================================================
def ask_choice(prompt, valid_options):
    """Ask for a value that must match one of the valid options (case-insensitive)."""
    options_str = ", ".join(valid_options)
    while True:
        value = input(f"{prompt} ({options_str}): ").strip()
        for opt in valid_options:
            if value.lower() == opt.lower():
                return opt
        print(f"  [!] Invalid entry. Please choose from: {options_str}\n")


def ask_binary(prompt):
    """Ask for a 0 or 1 value."""
    while True:
        value = input(f"{prompt} (0 = No, 1 = Yes): ").strip()
        if value in ('0', '1'):
            return int(value)
        print("  [!] Invalid entry. Please enter 0 or 1.\n")


def ask_number(prompt, min_val, max_val):
    """Ask for a numeric value within a valid range."""
    while True:
        value = input(f"{prompt} ({min_val} - {max_val}): ").strip()
        try:
            num = float(value)
            if min_val <= num <= max_val:
                return num
            print(f"  [!] Value must be between {min_val} and {max_val}.\n")
        except ValueError:
            print("  [!] Invalid entry. Please enter a number.\n")


# ============================================================
# INTERACTIVE INPUT — USER ENTERS PATIENT DETAILS
# ============================================================
if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  ENTER PATIENT DETAILS FOR PREDICTION")
    print("=" * 55)
    print("(Type the value and press Enter for each question)\n")

    # Collect each input with validation
    gender        = ask_choice("Gender", VALID_GENDER)
    age           = ask_number("Age (years)", 18, 120)
    hypertension  = ask_binary("Hypertension")
    heart_disease = ask_binary("Heart Disease")
    smoking       = ask_choice("Smoking History", VALID_SMOKING)
    bmi           = ask_number("BMI", 10, 60)
    hba1c         = ask_number("HbA1c Level", 3.0, 9.0)
    glucose       = ask_number("Blood Glucose Level", 50, 300)

    # Show a summary of the entered data
    print("\n" + "-" * 55)
    print("  PATIENT DATA ENTERED:")
    print("-" * 55)
    print(f"  Gender              : {gender}")
    print(f"  Age                 : {int(age)}")
    print(f"  Hypertension        : {'Yes' if hypertension else 'No'}")
    print(f"  Heart Disease       : {'Yes' if heart_disease else 'No'}")
    print(f"  Smoking History     : {smoking}")
    print(f"  BMI                 : {bmi}")
    print(f"  HbA1c Level         : {hba1c}")
    print(f"  Blood Glucose Level : {int(glucose)}")

    # Make the prediction
    result = predict_diabetes(gender, age, hypertension, heart_disease,
                              smoking, bmi, hba1c, glucose)

    # Show the result
    print("\n" + "=" * 55)
    print("  PREDICTION RESULT")
    print("=" * 55)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Probability : {result['probability']}%")
    print(f"  Risk Level  : {result['risk_level']}")
    print("=" * 55)
