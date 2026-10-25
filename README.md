# Heart Disease Prediction Dashboard

A Streamlit-based dashboard for training, comparing, and explaining machine learning models for heart disease prediction. The project is designed for both clinical and non-clinical staff to explore patient data, tune models, compare performance, and generate live predictions from new inputs.

## Overview

This application provides a complete workflow for heart disease classification:

- Upload a CSV dataset
- Select a target column and preprocess data
- Explore data distributions and relationships
- Train multiple classification models
- Tune model hyperparameters with GridSearchCV
- Compare model metrics and confusion matrices
- Evaluate feature importance and SHAP explanations
- Make live predictions using new patient values

## Features

- Interactive UI built with Streamlit
- Multiple ML models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - LightGBM
  - XGBoost
- Data preprocessing options:
  - Remove null values
  - Remove duplicates
  - Fill missing values
  - Encode categorical data
  - Scale numeric features
- Exploratory data analysis visualizations
- Train and compare model performance
- Hyperparameter tuning with GridSearchCV
- Confusion matrices and classification reports
- SHAP-based explainability
- Live patient prediction interface

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
├── app.py                     # Streamlit dashboard application
├── dataset_1.csv             # Sample heart disease dataset
├── dataset_2.csv             # Additional sample heart disease dataset
├── python_notebook.ipynb     # Jupyter notebook for analysis and experimentation
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules (if present in repo)
```

## Dataset

The dashboard supports CSV datasets containing medical and demographic variables commonly used for heart disease prediction, such as:

- age
- sex
- chest pain type (cp)
- resting blood pressure (trestbps)
- cholesterol (chol)
- fasting blood sugar (fbs)
- resting ECG results (restecg)
- maximum heart rate (thalach)
- exercise-induced angina (exang)
- oldpeak
- slope
- ca
- thal
- target

This project was built around the classic heart disease classification dataset structure and can work with similar tabular health datasets.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/khan-shayan/Heart-Disease-Prediction-dashboard.git
cd Heart-Disease-Prediction-dashboard
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # On macOS/Linux
.venv\Scripts\activate       # On Windows
```

3. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap
```

## Running the App

Start the dashboard with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal (typically http://localhost:8501).

## How to Use

1. Upload a CSV file from the sidebar.
2. Select the target column if needed.
3. Choose preprocessing steps such as removing nulls, filling missing values, encoding categoricals, and scaling features.
4. Explore the uploaded dataset using the visual analytics tabs.
5. Go to the Train & Tune section and select the models to train.
6. Run auto-tuning to find good parameter settings.
7. Train the selected models with custom hyperparameters.
8. Review model performance metrics, confusion matrices, and feature importance.
9. Use the Explanation tab to inspect SHAP summaries.
10. Use the Live Prediction tab to input patient values and predict heart disease risk.

## Model Comparison

The app evaluates multiple classifiers and compares them using metrics such as:

- Precision
- Recall
- F1-score
- Confusion matrix
- Weighted average metrics

## Explanation and Interpretability

SHAP is used to explain predictions and help understand how individual features contribute to the model's output. This is especially useful in healthcare applications where interpretability helps build trust and support decisions.

## Notes

- The dashboard is built for educational and demonstration purposes.
- Model performance depends on the quality and structure of the dataset.
- For best results, use a cleaned dataset with a clearly defined target column.

## License

This project does not currently specify a license in the repository. If you plan to distribute or reuse the code, consider adding an appropriate open-source license.

## Author

Khan Shayan

## Repository

https://github.com/khan-shayan/Heart-Disease-Prediction-dashboard
