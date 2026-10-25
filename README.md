# Heart Disease Prediction Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-LightGBM-9ACD32?style=for-the-badge" alt="XGBoost and LightGBM" />
</p>

A Streamlit-powered dashboard for heart disease prediction, built to help both clinical and non-clinical users explore data, train machine learning models, compare performance, and generate live predictions using patient attributes.

## Overview

This project provides an end-to-end machine learning workflow for identifying cardiovascular risk from tabular health data. It includes:

- Dataset upload and preprocessing
- Exploratory Data Analysis (EDA)
- Hyperparameter tuning
- Multi-model comparison
- Explainability with SHAP
- Real-time prediction from custom inputs

## Why this project?

Heart disease prediction is a critical healthcare task, and using machine learning can support early screening and better clinical decision support. This dashboard makes model training and interpretation more accessible by putting the complete workflow in a lightweight, interactive web app.

## Features

- Upload CSV datasets directly from the browser
- Select target columns and preprocess the data
- Clean missing values, remove duplicates, and encode categorical features
- Train multiple models with custom hyperparameters
- Run GridSearchCV-based tuning for improved performance
- Compare confusion matrices and classification reports
- View feature importance and SHAP explanations
- Generate live predictions with patient-level input values

## Supported Models

- Decision Tree
- Random Forest
- Gradient Boosting
- LightGBM
- XGBoost

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- LightGBM
- SHAP

## Project Structure

```text
Heart-Disease-Prediction-dashboard/
├── app.py
├── dataset_1.csv
├── dataset_2.csv
├── python_notebook.ipynb
├── README.md
└── .gitignore
```

## Dataset Requirements

The app is designed for tabular CSV data similar to standard heart disease datasets. Typical features include:

- age
- sex
- chest pain type
- resting blood pressure
- cholesterol
- fasting blood sugar
- resting ECG
- maximum heart rate
- exercise-induced angina
- oldpeak
- slope
- ca
- thal
- target

## Installation

1. Clone the repository:

```bash
git clone https://github.com/khan-shayan/Heart-Disease-Prediction-dashboard.git
cd Heart-Disease-Prediction-dashboard
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
```

3. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap
```

## Run the App

```bash
streamlit run app.py
```

After running the command, open the local URL shown in the terminal (usually http://localhost:8501).

## How to Use

1. Upload a CSV dataset in the sidebar.
2. Select the target column.
3. Apply preprocessing options such as null handling, encoding, and scaling.
4. Explore the dataset visually using the EDA tabs.
5. Choose the models you want to train.
6. Run tuning and train the models.
7. Compare model performance in the results section.
8. Use SHAP to understand predictions.
9. Enter patient values in the Live Prediction tab to estimate risk.

## Example Workflow

```text
Upload data → Preprocess → Explore → Train models → Tune hyperparameters → Evaluate → Explain → Predict
```

## Model Evaluation

The dashboard compares models using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Weighted average metrics

## Explainability

SHAP is integrated to help users interpret model decisions and understand which features influence heart disease risk predictions.

## Notes

- Performance depends on the dataset quality and selected features.
- Some datasets may require additional cleaning or feature engineering.
- This project is intended for learning, research, and demonstration purposes.

## License

This repository does not currently include a specified open-source license. If you plan to distribute or reuse the project commercially or publicly, consider adding an appropriate license file.

## Author

Khan Shayan

## Repository

https://github.com/khan-shayan/Heart-Disease-Prediction-dashboard
