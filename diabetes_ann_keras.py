#Refined version
# ============================================================
# Diabetes Risk Prediction — Artificial Neural Network (ANN)
# BAXI 3133 | Neural Network | Group Mini Project
# Technique   : Multilayer Perceptron (MLP)
# Framework   : TensorFlow / Keras
# Methodology : CRISP-DM
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("  DIABETES RISK PREDICTION — ANN (Keras / TensorFlow)")
print(f"  TensorFlow version : {tf.__version__}")
print("=" * 60)

# ============================================================
# PHASE 1 & 2 — BUSINESS & DATA UNDERSTANDING
# ============================================================
print("\n[PHASE 1-2] Business & Data Understanding")
print("-" * 40)

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

print(f"Dataset Shape      : {df.shape}")
print(f"Total Features     : {df.shape[1] - 1}")
print(f"Total Records      : {df.shape[0]}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())
print("\nClass Distribution:")
print(df['diabetes'].value_counts())
print(f"\nClass Imbalance Ratio: "
      f"{df['diabetes'].value_counts()[0] / df['diabetes'].value_counts()[1]:.1f}:1")

# ============================================================
# PHASE 3A — DATA CONSOLIDATION
# ============================================================
print("\n[PHASE 3A] Data Consolidation")
print("-" * 40)

print("Missing Values per Column:")
print(df.isnull().sum())
print(f"Total Missing   : {df.isnull().sum().sum()}")
print(f"Total Duplicates: {df.duplicated().sum()}")

# ============================================================
# PHASE 3B — DATA CLEANING
# ============================================================
print("\n[PHASE 3B] Data Cleaning")
print("-" * 40)

rows_before = len(df)

# Step 1: Remove duplicate rows
df = df.drop_duplicates()
print(f"Duplicates removed    : {rows_before - len(df)} rows")

# Step 2: Remove invalid gender entries
df = df[df['gender'] != 'Other']
print(f"Removed 'Other' gender: kept {len(df)} rows")

# Step 3: Keep only adults (age >= 18)
# Type 2 diabetes is predominantly an adult disease, and the clinical
# indicators in this dataset (HbA1c, BMI, glucose) are adult risk markers.
# This filter also removes the unrealistic decimal/infant ages (0.08 - 1.88).
df = df[df['age'] >= 18]
print(f"After age filter (>=18): {len(df)} rows")

# Step 4: Remove outliers using IQR method on continuous features
def remove_outliers_iqr(dataframe, column, multiplier=3.0):
    Q1    = dataframe[column].quantile(0.25)
    Q3    = dataframe[column].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    before = len(dataframe)
    dataframe = dataframe[
        (dataframe[column] >= lower) & (dataframe[column] <= upper)
    ]
    print(f"  {column:<25}: removed {before - len(dataframe)} outliers "
          f"(valid range: {lower:.2f} - {upper:.2f})")
    return dataframe

print("Outlier removal (IQR x3):")
for col in ['bmi', 'blood_glucose_level', 'HbA1c_level']:
    df = remove_outliers_iqr(df, col)

print(f"\nFinal Clean Dataset   : {df.shape[0]:,} rows "
      f"({rows_before - df.shape[0]} removed total)")

# ============================================================
# PHASE 3C — DATA TRANSFORMATION
# ============================================================
print("\n[PHASE 3C] Data Transformation")
print("-" * 40)

# Label encode categorical columns
le_gender  = LabelEncoder()
le_smoking = LabelEncoder()

df['gender']          = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

print("Label Encoding applied:")
print(f"  gender         : {list(le_gender.classes_)} "
      f"-> {list(range(len(le_gender.classes_)))}")
print(f"  smoking_history: {list(le_smoking.classes_)} "
      f"-> {list(range(len(le_smoking.classes_)))}")

# ============================================================
# PHASE 3D — DATA REDUCTION (Correlation Heatmap)
# ============================================================
print("\n[PHASE 3D] Data Reduction — Correlation Analysis")
print("-" * 40)

# Calculate correlation of all features with target
X_all = df.drop('diabetes', axis=1)
y_all = df['diabetes']

correlations = X_all.corrwith(y_all).abs().sort_values(ascending=False)
print("Feature Correlation with Target (absolute):")
for feat, corr in correlations.items():
    bar = '█' * int(corr * 30)
    print(f"  {feat:<25} {corr:.4f}  {bar}")

feature_names = X_all.columns.tolist()
print("\nDecision: All 8 features retained (all show meaningful correlation > 0.01)")

# ============================================================
# PHASE 4 — MODELLING (ANN with Keras)
# ============================================================
print("\n[PHASE 4] Modelling — ANN Architecture (Keras/TensorFlow)")
print("-" * 40)

X = df.drop('diabetes', axis=1)
y = df['diabetes']

# --- Train / Validation / Test Split (70 / 15 / 15) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train : {X_train.shape[0]:,} samples")
print(f"Val   : {X_val.shape[0]:,} samples")
print(f"Test  : {X_test.shape[0]:,} samples")

# --- Feature Scaling ---
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# --- Handle Class Imbalance using class_weight ---
# This tells the model to pay more attention to minority class (diabetic)
classes = np.unique(y_train)
cw      = class_weight.compute_class_weight(
              'balanced', classes=classes, y=y_train)
cw_dict = dict(zip(classes, cw))
print(f"\nClass Weights (balanced): {cw_dict}")

# --- Build ANN Model using Keras ---
# Each layer is explicitly defined to show the ANN architecture clearly
print("\n--- ANN Architecture ---")

model = Sequential([
    # Input layer — accepts 8 features
    Input(shape=(X_train_s.shape[1],), name='Input_Layer'),

    # Hidden Layer 1 — 128 neurons, ReLU activation
    Dense(128, activation='relu',
          kernel_regularizer=l2(0.001),
          name='Hidden_Layer_1'),
    Dropout(0.3, name='Dropout_1'),  # Randomly drop 30% neurons to prevent overfitting

    # Hidden Layer 2 — 64 neurons, ReLU activation
    Dense(64, activation='relu',
          kernel_regularizer=l2(0.001),
          name='Hidden_Layer_2'),
    Dropout(0.2, name='Dropout_2'),  # Randomly drop 20% neurons

    # Hidden Layer 3 — 32 neurons, ReLU activation
    Dense(32, activation='relu',
          kernel_regularizer=l2(0.001),
          name='Hidden_Layer_3'),
    Dropout(0.2, name='Dropout_3'),

    # Output Layer — 1 neuron, Sigmoid activation for binary classification
    # Sigmoid outputs probability between 0 and 1
    Dense(1, activation='sigmoid', name='Output_Layer')
], name='ANN_Diabetes_Predictor')

# --- Compile the ANN ---
# Adam optimizer adjusts learning rate automatically
# BinaryCrossentropy is the standard loss function for binary classification
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Print model summary — shows every layer and number of parameters
model.summary()

# --- Early Stopping Callback ---
# Stops training automatically when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# --- Train the ANN ---
print("\n--- Training ANN ---")
history = model.fit(
    X_train_s, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val_s, y_val),
    class_weight=cw_dict,   # Handle class imbalance
    callbacks=[early_stopping],
    verbose=1
)

print(f"\nTraining complete! "
      f"Stopped at epoch {len(history.history['loss'])} / 100")

# ============================================================
# PHASE 4B — STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================
print("\n[PHASE 4B] Stratified K-Fold Cross Validation (5 Folds)")
print("-" * 40)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
cv_auc = []

X_np = X_train_s
y_np = y_train.values

fold = 1
for train_idx, val_idx in skf.split(X_np, y_np):
    Xf_train, Xf_val = X_np[train_idx], X_np[val_idx]
    yf_train, yf_val = y_np[train_idx], y_np[val_idx]

    # Build a smaller model for each fold
    fold_model = Sequential([
        Input(shape=(Xf_train.shape[1],)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(64,  activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(32,  activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1,   activation='sigmoid')
    ])
    fold_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    fold_model.fit(
        Xf_train, yf_train,
        epochs=50,
        batch_size=64,
        validation_data=(Xf_val, yf_val),
        class_weight=cw_dict,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                 restore_best_weights=True)],
        verbose=0
    )

    yf_pred  = (fold_model.predict(Xf_val, verbose=0) > 0.5).astype(int).flatten()
    yf_proba = fold_model.predict(Xf_val, verbose=0).flatten()

    acc = accuracy_score(yf_val, yf_pred)
    auc = roc_auc_score(yf_val, yf_proba)
    cv_acc.append(acc)
    cv_auc.append(auc)
    print(f"  Fold {fold}: Accuracy = {acc:.4f} | ROC-AUC = {auc:.4f}")
    fold += 1

print(f"\nCross Validation Results (5-Fold):")
print(f"  Mean Accuracy : {np.mean(cv_acc):.4f} (+/- {np.std(cv_acc):.4f})")
print(f"  Mean ROC-AUC  : {np.mean(cv_auc):.4f} (+/- {np.std(cv_auc):.4f})")

# ============================================================
# PHASE 5 — EVALUATION
# ============================================================
print("\n[PHASE 5] Evaluation")
print("-" * 40)

# Validation set predictions
y_val_proba = model.predict(X_val_s, verbose=0).flatten()
y_val_pred  = (y_val_proba > 0.5).astype(int)
print("--- Validation Set ---")
print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_val, y_val_proba):.4f}")

# Test set predictions
y_test_proba = model.predict(X_test_s, verbose=0).flatten()
y_test_pred  = (y_test_proba > 0.5).astype(int)
print("\n--- Test Set ---")
print(f"Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_test_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred,
      target_names=['No Diabetes', 'Diabetes']))

# --- Permutation Feature Importance (Custom for Keras) ---
# Shuffles each feature one at a time and measures how much ROC-AUC drops
# A bigger drop = that feature is more important to the model
print("--- Permutation Feature Importance ---")

def keras_permutation_importance(model, X, y, feature_names, n_repeats=10):
    """
    Custom permutation importance for Keras models.
    For each feature: shuffle its values randomly and measure
    how much the ROC-AUC score drops compared to baseline.
    """
    # Get baseline score with original data
    baseline = roc_auc_score(y, model.predict(X, verbose=0).flatten())
    importances = []
    for i in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])  # Shuffle one feature
            score = roc_auc_score(y, model.predict(X_permuted, verbose=0).flatten())
            scores.append(baseline - score)      # Drop in performance
        importances.append(np.mean(scores))
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

feat_imp = keras_permutation_importance(
    model, X_test_s, y_test, feature_names, n_repeats=10
)
print(feat_imp.to_string())

# ============================================================
# PHASE 6 — DEPLOYMENT
# ============================================================
print("\n[PHASE 6] Deployment")
print("-" * 40)

# Save the trained model and preprocessing objects
model.save('diabetes_ann_model.keras')
joblib.dump(scaler,     'diabetes_scaler.pkl')
joblib.dump(le_gender,  'diabetes_le_gender.pkl')
joblib.dump(le_smoking, 'diabetes_le_smoking.pkl')
print("ANN model saved  : diabetes_ann_model.keras")
print("Scaler saved     : diabetes_scaler.pkl")

# Prediction function — use this to predict for any new patient
def predict_diabetes(gender, age, hypertension, heart_disease,
                     smoking_history, bmi, HbA1c_level, blood_glucose_level):
    """
    Predict diabetes risk for a new patient.

    Parameters:
        gender              : 'Male' or 'Female'
        age                 : float (years)
        hypertension        : 0 = No, 1 = Yes
        heart_disease       : 0 = No, 1 = Yes
        smoking_history     : 'never', 'former', 'current',
                              'not current', 'ever', 'No Info'
        bmi                 : float
        HbA1c_level         : float (normal <5.7, diabetic >=6.5)
        blood_glucose_level : int   (normal <100, diabetic >=126)

    Returns:
        Dictionary with prediction, probability and risk level
    """
    from tensorflow.keras.models import load_model

    ann  = load_model('diabetes_ann_model.keras')
    scl  = joblib.load('diabetes_scaler.pkl')
    le_g = joblib.load('diabetes_le_gender.pkl')
    le_s = joblib.load('diabetes_le_smoking.pkl')

    gender_enc  = le_g.transform([gender])[0]
    smoking_enc = le_s.transform([smoking_history])[0]

    features        = np.array([[gender_enc, age, hypertension, heart_disease,
                                  smoking_enc, bmi, HbA1c_level,
                                  blood_glucose_level]])
    features_scaled = scl.transform(features)

    proba = float(ann.predict(features_scaled, verbose=0)[0][0])

    # Cap at 99% — no model should ever claim 100% certainty
    proba_display = min(round(proba * 100, 2), 99.0)

    return {
        'prediction'  : 'Diabetic' if proba > 0.5 else 'Not Diabetic',
        'probability' : proba_display,
        'risk_level'  : 'HIGH'   if proba > 0.70 else
                        'MEDIUM' if proba > 0.40 else 'LOW'
    }

# Demo predictions — test with 4 different patient profiles
print("\n--- Demo Predictions (Multiple Patients) ---")
demo_patients = [
    ("Patient A (Low risk)",    'Female', 25, 0, 0, 'never',   21.5, 4.8,  85),
    ("Patient B (Medium risk)", 'Male',   48, 0, 0, 'former',  28.0, 5.6, 130),
    ("Patient C (High risk)",   'Male',   62, 1, 1, 'current', 33.0, 6.3, 172),
    ("Patient D (Very high)",   'Female', 57, 1, 0, 'former',  31.5, 8.5, 235),
]

print(f"\n{'Patient':<30} {'Prediction':<15} {'Probability':>12}  {'Risk'}")
print("-" * 65)
for label, *params in demo_patients:
    r = predict_diabetes(*params)
    print(f"{label:<30} {r['prediction']:<15} "
          f"{r['probability']:>10.2f}%  {r['risk_level']}")

# ============================================================
# VISUALIZATIONS (7 PLOTS)
# ============================================================
print("\n--- Generating Visualizations ---")

fig = plt.figure(figsize=(22, 14))
fig.suptitle(
    'ANN Diabetes Risk Prediction — Full CRISP-DM Analysis\n'
    '(Multilayer Perceptron | TensorFlow / Keras)',
    fontsize=15, fontweight='bold', y=0.98
)

# --- Plot 1: Class Distribution ---
ax1 = fig.add_subplot(2, 4, 1)
counts = df['diabetes'].value_counts()
bars   = ax1.bar(['No Diabetes', 'Diabetes'], counts.values,
                  color=['#2196F3', '#F44336'], edgecolor='white', width=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,}\n({val/len(df)*100:.1f}%)',
             ha='center', fontsize=9, fontweight='bold')
ax1.set_title('Class Distribution', fontweight='bold')
ax1.set_ylabel('Count')
ax1.grid(axis='y', alpha=0.3)

# --- Plot 2: Correlation Heatmap ---
ax2 = fig.add_subplot(2, 4, 2)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='Blues',
            ax=ax2, linewidths=0.5, square=False)
ax2.set_title('Correlation Heatmap', fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=7)
ax2.tick_params(axis='y', rotation=0, labelsize=7)

# --- Plot 3: Feature Correlation with Target ---
ax3 = fig.add_subplot(2, 4, 3)
colors3 = ['#1565C0' if c > 0.2 else '#42A5F5' for c in correlations.values]
correlations.plot(kind='barh', ax=ax3, color=colors3, edgecolor='white')
ax3.set_title('Feature Correlation\nwith Diabetes', fontweight='bold')
ax3.set_xlabel('Absolute Correlation')
ax3.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='0.2 threshold')
ax3.legend(fontsize=8)
ax3.grid(axis='x', alpha=0.3)

# --- Plot 4: Training & Validation Loss ---
ax4 = fig.add_subplot(2, 4, 4)
ax4.plot(history.history['loss'],     color='steelblue',  lw=2, label='Train Loss')
ax4.plot(history.history['val_loss'], color='darkorange', lw=2,
         linestyle='--', label='Val Loss')
ax4.set_title('Training & Validation Loss', fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Binary Cross-Entropy Loss')
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- Plot 5: Training & Validation Accuracy ---
ax5 = fig.add_subplot(2, 4, 5)
ax5.plot(history.history['accuracy'],     color='steelblue',  lw=2,
         label='Train Accuracy')
ax5.plot(history.history['val_accuracy'], color='darkorange', lw=2,
         linestyle='--', label='Val Accuracy')
ax5.set_title('Training & Validation Accuracy', fontweight='bold')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Accuracy')
ax5.legend()
ax5.grid(True, alpha=0.3)

# --- Plot 6: Confusion Matrix ---
ax6 = fig.add_subplot(2, 4, 6)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            annot_kws={'size': 13, 'weight': 'bold'})
ax6.set_title('Confusion Matrix (Test Set)', fontweight='bold')
ax6.set_ylabel('Actual')
ax6.set_xlabel('Predicted')

# --- Plot 7: ROC Curve ---
ax7 = fig.add_subplot(2, 4, 7)
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auc_score   = roc_auc_score(y_test, y_test_proba)
ax7.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {auc_score:.4f})')
ax7.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
ax7.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
ax7.set_title('ROC Curve (Test Set)', fontweight='bold')
ax7.set_xlabel('False Positive Rate')
ax7.set_ylabel('True Positive Rate')
ax7.legend(loc='lower right', fontsize=8)
ax7.grid(True, alpha=0.3)

# --- Plot 8: Permutation Feature Importance ---
ax8 = fig.add_subplot(2, 4, 8)
feat_imp_sorted = feat_imp.sort_values()
colors8 = ['#1565C0' if v > 0.05 else '#90CAF9' for v in feat_imp_sorted.values]
feat_imp_sorted.plot(kind='barh', ax=ax8, color=colors8, edgecolor='white')
ax8.set_title('Permutation Feature Importance\n(ROC-AUC drop)', fontweight='bold')
ax8.set_xlabel('Mean Importance')
ax8.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ann_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plots saved as 'ann_results.png'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Framework                : TensorFlow / Keras {tf.__version__}")
print(f"  Technique                : Multilayer Perceptron (MLP)")
print(f"  Dataset (after cleaning) : {df.shape[0]:,} records")
print(f"  Features used            : {X.shape[1]}")
print(f"  ANN Architecture         : 8 -> 128 -> 64 -> 32 -> 1")
print(f"  Activation (hidden)      : ReLU")
print(f"  Activation (output)      : Sigmoid")
print(f"  Loss Function            : Binary Cross-Entropy")
print(f"  Optimiser                : Adam (lr=0.001)")
print(f"  Training Epochs          : {len(history.history['loss'])} / 100")
print(f"  CV Accuracy (5-Fold)     : {np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f}")
print(f"  CV ROC-AUC  (5-Fold)     : {np.mean(cv_auc):.4f} +/- {np.std(cv_auc):.4f}")
print(f"  Test Accuracy            : {accuracy_score(y_test, y_test_pred)*100:.2f}%")
print(f"  Test ROC-AUC             : {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"  Model file               : diabetes_ann_model.keras")
print("=" * 60)
