# Diabetes Risk Prediction — ANN (MLP) Setup Guide
## BAXI 3133 | Neural Network | Group Mini Project

---

## 📁 What You Will Get After Downloading

```
📂 Your Project Folder
 ├── diabetes_ann_keras.py            ← Training code
 ├── predict.py                       ← Prediction script
 ├── diabetes_prediction_dataset.csv  ← Dataset
 ├── README.md                        ← Setup guide
 │
 ├── diabetes_ann_model.keras         ← Saved model
 ├── diabetes_scaler.pkl              ← Saved scaler
 ├── diabetes_le_gender.pkl           ← Gender encoder
 ├── diabetes_le_smoking.pkl          ← Smoking encoder
 │
 ├── ann_results.png                  ← Sample output graph ✅
 └── model_results.txt                ← Sample metrics ✅
```

---

## 💻 Step 1 — Install Python

1. Go to **https://www.python.org/downloads/**
2. Download **Python 3.10 or above**
3. During installation — **tick the box "Add Python to PATH"**
4. Click Install Now

To verify Python is installed, open **Command Prompt** and type:
```
python --version
```
You should see something like: `Python 3.10.x`

---

## 📥 Step 2 — Download Project from GitHub

### Option A — Download as ZIP (Easier)
1. Go to the GitHub repository link shared by your groupmate
2. Click the green **Code** button
3. Click **Download ZIP**
4. Extract the ZIP file to any folder on your computer

### Option B — Clone using Git
If you have Git installed, open Command Prompt and run:
```
git clone https://github.com/YOUR_REPO_LINK_HERE.git
```

---

## 📦 Step 3 — Install Required Libraries

Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux).

Navigate to your project folder:
```
cd path/to/your/project/folder
```

Install all required libraries with this single command:
```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

Or install one by one if the above does not work:
```
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tensorflow
pip install joblib
```

---

## ✅ Step 4 — Verify Installation

After installing, verify everything works by running this in Command Prompt:
```
python -c "import pandas; import numpy; import matplotlib; import seaborn; import sklearn; import tensorflow; import joblib; print('All libraries installed successfully!')"
```

If you see **"All libraries installed successfully!"** you are good to go!

---

## ▶️ Step 5 — Run the Project

There are **two ways** to use this project:

### Option A — Train the Model from Scratch
This runs the full pipeline (data cleaning, training, evaluation, plots).
Make sure `diabetes_prediction_dataset.csv` is in the same folder, then run:
```
python diabetes_ann_keras.py
```
⏱️ Takes about 10–20 minutes (see timing below).

### Option B — Make Predictions Instantly (No Training)
If the saved model files are already present, you can run predictions
straight away **without retraining**. This loads the trained model and
asks you to enter patient details:
```
python predict.py
```
✅ This gives identical results every time and only takes a few seconds.

---

## ⏱️ Expected Running Time (Option A — Training)

| Step | Estimated Time |
|---|---|
| Data Loading & Cleaning | Less than 1 minute |
| ANN Training (100 epochs max) | 2 to 5 minutes |
| Stratified K-Fold (5 folds) | 5 to 10 minutes |
| Permutation Feature Importance | 3 to 5 minutes |
| Generating Plots | Less than 1 minute |
| **Total** | **10 to 20 minutes** |

> ⚠️ Be patient during K-Fold and Permutation Importance steps.
> The program is still running even if it looks like nothing is happening.

---

## 📤 Output Files Generated (After Training)

After running `diabetes_ann_keras.py` successfully, these files appear:

```
📂 Your Project Folder
 ├── diabetes_ann_model.keras    ← Saved trained ANN model
 ├── diabetes_scaler.pkl         ← Saved StandardScaler
 ├── diabetes_le_gender.pkl      ← Saved gender encoder
 ├── diabetes_le_smoking.pkl     ← Saved smoking encoder
 ├── ann_results.png             ← All visualisation plots
 └── model_results.txt           ← All metrics (accuracy, ROC-AUC, etc.)
```

---

## 🧪 Using predict.py (Interactive Prediction)

When you run `python predict.py`, it will ask you to enter patient details
one by one. It validates each input, so wrong entries are caught.

Example:
```
Gender (Female, Male): Male
Age (years) (18 - 120): 55
Hypertension (0 = No, 1 = Yes): 1
Heart Disease (0 = No, 1 = Yes): 0
Smoking History (No Info, current, ever, former, never, not current): former
BMI (10 - 60): 30.0
HbA1c Level (3.0 - 9.0): 6.5
Blood Glucose Level (50 - 300): 160
```

It then displays the prediction, probability, and risk level.

> 📌 Note: `predict.py` needs the 4 saved files
> (`diabetes_ann_model.keras`, `diabetes_scaler.pkl`,
> `diabetes_le_gender.pkl`, `diabetes_le_smoking.pkl`) in the same folder.
> Run it in a terminal / Command Prompt so you can type your answers.

---

## 📋 Full Library List

| Library | Purpose |
|---|---|
| `pandas` | Load and manipulate dataset |
| `numpy` | Numerical computations |
| `matplotlib` | Plot graphs and charts |
| `seaborn` | Statistical visualisation (heatmap) |
| `scikit-learn` | Preprocessing, metrics, cross validation |
| `tensorflow` | Build and train the ANN model (Keras) |
| `joblib` | Save and load model files |

---

## ❗ Common Problems & Solutions

### Problem 1: `ModuleNotFoundError`
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution:** Run `pip install tensorflow` again

---

### Problem 2: `FileNotFoundError`
```
FileNotFoundError: diabetes_prediction_dataset.csv not found
```
**Solution:** Make sure the CSV file is in the **same folder** as the Python file

---

### Problem 3: Python not recognized
```
'python' is not recognized as an internal or external command
```
**Solution:** Re-install Python and make sure you **tick "Add Python to PATH"** during installation

---

### Problem 4: pip not working
```
'pip' is not recognized
```
**Solution:** Try using `pip3` instead:
```
pip3 install tensorflow
```

---

### Problem 5: TensorFlow installation takes too long
**Solution:** This is normal. TensorFlow is a large library (~500MB). Just wait for it to finish.

---

### Problem 6: GPU warning when running
```
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows...
```
**Solution:** This is harmless. It just means TensorFlow uses the CPU instead of GPU.
The code works perfectly fine — you can ignore this message.

---

### Problem 7: predict.py says model file not found
```
No file or directory: 'diabetes_ann_model.keras'
```
**Solution:** Either run `diabetes_ann_keras.py` first to create the model,
or make sure the 4 saved files are in the same folder as `predict.py`.

---

## 💡 Recommended IDE (Optional)

You can run the code using any of these:

| IDE | How to get it |
|---|---|
| **VS Code** (Recommended) | https://code.visualstudio.com/ |
| **Jupyter Notebook** | Run `pip install notebook` then `jupyter notebook` |
| **PyCharm** | https://www.jetbrains.com/pycharm/ |
| **Command Prompt** | Already available on Windows — no install needed |

> 📌 For `predict.py`, use a terminal / Command Prompt (or the VS Code terminal),
> since it needs keyboard input.

---

## 🐍 Recommended Python Version

| | Recommended |
|---|---|
| Python | 3.10 or 3.11 |
| TensorFlow | 2.13 or above |
| scikit-learn | 1.3 or above |

---

## ♻️ Note on Reproducibility

A random seed (42) is set in the code for reproducibility, so results are
consistent on the same machine. Minor variation (less than 1%) may occur on
different hardware due to floating-point operations — this is normal for
deep learning. For exactly identical results, use `predict.py`, which loads
the saved model without retraining.

---

## 👥 Project Information

| | |
|---|---|
| **Subject** | BAXI 3133 Neural Network |
| **Project** | Diabetes Risk Prediction |
| **Technique** | Multilayer Perceptron (MLP) |
| **Framework** | TensorFlow / Keras |
| **Dataset** | Kaggle Diabetes Prediction Dataset |
| **Supervised by** | Prof Ts Dr Burhanuddin, UTeM |

---
