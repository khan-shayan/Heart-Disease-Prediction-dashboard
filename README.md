# Heart-Disease-Prediction-dashboard

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-9ACD32?style=for-the-badge)](https://xgboost.readthedocs.io/en/stable/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Official%20Docs-2E8B57?style=for-the-badge)](https://lightgbm.readthedocs.io/en/stable/)

Streamlit-based dashboard for both clinical and non-clinical staff, featuring live tree-based model training, hyperparameter tuning, model comparison, and live heart disease prediction.

## Overview

This project is a machine learning dashboard built with Streamlit that helps users upload a heart disease dataset, preprocess it, train multiple classification models, compare their performance, and make predictions on new patient data.

It is designed for both technical and non-technical users, with a simple interface that supports exploratory data analysis, model tuning, and explainability.

## Features

- CSV dataset upload from the sidebar
- Preprocessing tools for null handling, duplicates, categorical encoding, and scaling
- EDA visualizations for continuous and categorical features
- Model tuning using GridSearchCV
- Comparison of multiple classifiers
- Confusion matrix and classification metrics
- Feature importance analysis
- SHAP-based explainability
- Live prediction form for patient-level input

## Supported Models

The dashboard supports the following models:

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
├── app.py                    # Streamlit application
├── dataset_1.csv             # Sample heart disease dataset
├── dataset_2.csv             # Additional sample dataset
├── python_notebook.ipynb     # Notebook used for analysis and prototyping
├── README.md                 # Project documentation
├── .gitignore                # Git ignore file
└── requirements.txt          # Optional dependency list if added later
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or newer
- pip installed
- A virtual environment is recommended but not required

## Installation

1. Clone the repository:

```bash
git clone https://github.com/khan-shayan/Heart-Disease-Prediction-dashboard.git
cd Heart-Disease-Prediction-dashboard
```

2. Create and activate a virtual environment:

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap
```

If you prefer a requirements file, you can also install from a generated list:

```bash
pip install -r requirements.txt
```

## Running the App

Start the dashboard with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use the Dashboard

1. Upload a CSV dataset containing heart disease data.
2. Select the target column for prediction.
3. Choose preprocessing options such as:
   - removing null values
   - removing duplicates
   - filling missing entries
   - categorical encoding
   - feature scaling
4. Explore charts and distributions in the data analysis section.
5. Go to the Train & Tune tab and select models.
6. Run hyperparameter tuning and train the chosen models.
7. Review the training results, confusion matrices, and performance metrics.
8. Use the Explanation tab to inspect SHAP values.
9. In Live Prediction, input patient features and generate a prediction.

## Data Format

The project expects the dataset to be in CSV format with tabular features and a target column (for example, a binary label where 0 = no disease and 1 = disease present).

Typical columns may include:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal
- target

## Model Training Workflow

The app supports the following workflow:

```text
Upload data -> preprocess data -> explore features -> tune hyperparameters -> train models -> compare results -> explain decisions -> make live predictions
```

## Evaluation Metrics

The dashboard compares models using classification reports and confusion matrices, focusing on metrics such as:

- accuracy
- precision
- recall
- F1-score
- weighted averages

## Explainability

The project includes SHAP-based explainability to help users understand why a model made a particular prediction. This is especially useful for health-related use cases where interpretability matters.

## Notes

- The app is intended for educational and research use.
- Model performance depends on the quality and structure of the dataset.
- Cleaning and feature selection may be necessary for higher accuracy.
- For real-world clinical use, the model should be tested and validated carefully by domain experts.

## Troubleshooting

If the app does not start:

- verify Python is installed correctly
- check that all dependencies are installed
- ensure the current working directory is the project folder
- try restarting the Streamlit server

If a model does not train:

- confirm the target column exists
- remove invalid or missing values in the selected data
- check that the dataset is properly encoded for machine learning

## License

This repository currently does not include a license file. If you plan to share or reuse this project publicly, consider adding an appropriate open-source license.

## Author

Khan Shayan

## Repository

https://github.com/khan-shayan/Heart-Disease-Prediction-dashboard
