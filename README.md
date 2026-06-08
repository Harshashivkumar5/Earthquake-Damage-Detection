# 🌍 Earthquake Damage Prediction

## 📌 Project Overview

Earthquake Damage Prediction is a Machine Learning project that predicts the level of structural damage to buildings after an earthquake. The model analyzes building characteristics, construction materials, geographical information, and structural attributes to classify the expected damage level.

This project aims to assist disaster management authorities, urban planners, and researchers in identifying vulnerable buildings and improving disaster preparedness strategies.

---

## 🎯 Objectives

* Predict building damage severity after an earthquake.
* Identify important factors affecting structural damage.
* Develop a reliable machine learning model for damage classification.
* Provide an interactive web application for real-time predictions.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
Earthquake_Damage_Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebooks/
│   ├── Earthquake_damage_detection.ipynb
│
├── outputs/
│   ├── visualizations
│
├── streamlit_app/
│   ├── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset Information

The dataset contains information about buildings affected by earthquakes, including:

* Building Age
* Building Height
* Foundation Type
* Roof Type
* Ground Condition
* Construction Material
* Geographic Region
* Structural Design Features

### Target Variable

Damage Grade:

| Grade | Description     |
| ----- | --------------- |
| 1     | Low Damage      |
| 2     | Moderate Damage |
| 3     | Severe Damage   |

---

## 🔍 Machine Learning Workflow

### 1. Data Preprocessing

* Missing value handling
* Data cleaning
* Feature encoding
* Feature scaling

### 2. Exploratory Data Analysis

* Distribution analysis
* Correlation analysis
* Damage pattern visualization

### 3. Feature Engineering

* Creation of derived features
* Selection of important predictors

### 4. Model Training

The following algorithms were evaluated:

* Decision Tree
* Random Forest
* XGBoost
* CatBoost

### 5. Model Evaluation

Evaluation Metrics:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Harshashivkumar5/Earthquake-Damage-Detection.git
```

Navigate to the project folder:

```bash
cd Earthquake-Damage-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/Earthquake_damage_detection.ipynb
```

### Run Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

---

## 📈 Results

The trained model successfully predicts building damage categories based on structural and environmental features.

Key benefits include:

* Faster damage assessment
* Improved disaster response planning
* Better resource allocation during emergencies

---

## ⚠️ Note

Large trained model files (.pkl) are not included in this repository due to GitHub file size limitations.

To generate the model file:

1. Open the notebook.
2. Run all cells.
3. Train the model.
4. Save the model locally using Joblib.

---

## 🔮 Future Enhancements

* Deep Learning-based prediction models
* Real-time earthquake data integration
* Model explainability using SHAP
* Cloud deployment
* REST API integration

---

## 👨‍💻 Author

**Harsha S**

* Information Science Graduate
* Data Analyst & Machine Learning Enthusiast

GitHub: https://github.com/Harshashivkumar5

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
