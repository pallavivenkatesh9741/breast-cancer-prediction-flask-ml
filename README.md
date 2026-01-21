Breast Cancer Prediction System using Machine Learning
📌 Project Overview

The Breast Cancer Prediction System is a machine learning–based web application developed using Flask and a Support Vector Machine (SVM) Classifier.
It predicts the likelihood of breast cancer based on key clinical attributes provided by the user through a web interface.

This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to deployment as a web application.

🎯 Objective

The objective of this project is to:

Predict the risk of breast cancer using clinical features

Assist in early detection and risk assessment

Showcase practical deployment of ML models using Flask

🧠 Machine Learning Model

Algorithm: Support Vector Machine (SVM) Classifier

Reason for Selection:

Effective in high-dimensional spaces

Handles binary classification well

Provides probability estimates for risk assessment

Model Evaluation Metrics:

Accuracy Score

Confusion Matrix

Classification Report

Probability-based prediction for risk level

📊 Dataset Description

The dataset is the Breast Cancer dataset from scikit-learn, containing multiple clinical and cellular attributes commonly used for breast cancer diagnosis.

Selected Features (used in the app):

mean radius – Mean of distances from center to points on the perimeter

mean texture – Standard deviation of gray-scale values

mean perimeter – Mean of perimeter measurements

mean area – Mean of area measurements

mean smoothness – Mean of local smoothness of the cell nuclei

Target Variable:

target

0 → Malignant (High Risk of Breast Cancer)

1 → Benign (Low Risk of Breast Cancer)

⚙️ Technology Stack

Language: Python

Web Framework: Flask

Machine Learning: Scikit-learn (SVM with GridSearchCV)

Data Handling: NumPy, Pandas

Frontend: HTML (Jinja2 Templates)

Model: Support Vector Machine (SVM)

🏗️ Project Structure
├── app.py
├── README.md
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    └── style.css (optional)
