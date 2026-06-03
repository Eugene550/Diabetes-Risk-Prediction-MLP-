# Diabetes Risk Prediction — ANN (MLP) Setup Guide
## BAXI 3133 | Neural Network | Group Mini Project

---

## 📁 Required Files

Make sure you have these files in the **same folder**:

```
📂 Your Project Folder
 ├── diabetes_ann_keras.py          ← Main Python code
 ├── diabetes_prediction_dataset.csv ← Dataset (download from Kaggle)
 └── README.md                       ← This file
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

## 📦 Step 2 — Install Required Libraries

Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux).

Navigate to your project folder:
```
cd path/to/your/project/folder
```

Then install all required libraries one by one:

```
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tensorflow
pip install joblib
```

Or install everything at once with this single command:
```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

---

## ✅ Step 3 — Verify Installation

After installing, verify everything works by running this in Command Prompt:

```
python -c "import pandas; import numpy; import matplotlib; import seaborn; import sklearn; import tensorflow; import joblib; print('All libraries installed successfully!')"
```

If you see **"All libraries installed successfully!"** you are good to go!

---

## 📊 Step 4 — Download the Dataset

1. Go to **https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset**
2. Sign in to Kaggle (create a free account if needed)
3. Click the **Download** button
4. Extract the ZIP file
5. Copy **diabetes_prediction_dataset.csv** into your project folder

---

## ▶️ Step 5 — Run the Code

Make sure your project folder contains:
- `diabetes_ann_keras.py`
- `diabetes_prediction_dataset.csv`

Then run the code:
```
python diabetes_ann_keras.py
```

---

## ⏱️ Expected Running Time

| Step | Estimated Time |
|---|---|
| Data Loading & Cleaning | Less than 1 minute |
| ANN Training (100 epochs max) | 2 to 5 minutes |
| Stratified K-Fold (5 folds) | 5 to 10 minutes |
| Permutation Feature Importance | 3 to 5 minutes |
| Generating Plots | Less than 1 minute |
| **Total** | **10 to 20 minutes** |

> ⚠️ Running time depends on your computer speed. Be patient during K-Fold and Permutation Importance steps.

---

## 📤 Output Files Generated

After running successfully, these files will be created in your project folder:

```
📂 Your Project Folder
 ├── diabetes_ann_model.keras    ← Saved trained ANN model
 ├── diabetes_scaler.pkl         ← Saved StandardScaler
 ├── diabetes_le_gender.pkl      ← Saved gender encoder
 ├── diabetes_le_smoking.pkl     ← Saved smoking encoder
 └── ann_results.png             ← All 8 visualisation plots
```

---

## 🐍 Recommended Python Version

| | Recommended |
|---|---|
| Python | 3.10 or 3.11 |
| TensorFlow | 2.13 or above |
| scikit-learn | 1.3 or above |

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

## 💡 Recommended IDE (Optional)

You can run the code using any of these:

| IDE | Download |
|---|---|
| **VS Code** (Recommended) | https://code.visualstudio.com/ |
| **Jupyter Notebook** | `pip install notebook` then run `jupyter notebook` |
| **PyCharm** | https://www.jetbrains.com/pycharm/ |
| **Command Prompt** | Already available on Windows |

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

> 📌 If you face any other problems, discuss with your groupmates or refer to the project report.
