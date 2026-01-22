##  Breast Cancer Prediction System using Machine Learning

### 📌 Project Overview

The Breast Cancer Prediction System is a machine learning–based web application developed using Flask and a Support Vector Machine (SVM) classifier.
It predicts the likelihood of breast cancer (benign or malignant) based on important diagnostic features entered by the user through a web interface.

This project demonstrates an end-to-end machine learning workflow, including data preprocessing, hyperparameter tuning, model training, and deployment using Flask.
___

### 🎯 Objective

The objective of this project is to:

Predict the risk of breast cancer using medical diagnostic measurements

Assist in early detection and awareness

Demonstrate the deployment of an optimized ML model using Flask
___

### 🧠 Machine Learning Model

Algorithm: Support Vector Machine (SVM)

Optimization: GridSearchCV with Stratified K-Fold Cross Validation

#### Why SVM?

Effective for high-dimensional data

Performs well on medical classification problems

Robust decision boundaries

#### Model Evaluation Techniques:

Stratified K-Fold Cross Validation

Probability-based prediction

Hyperparameter tuning (C, gamma, kernel)
___

###  📊 Dataset Description

The dataset is loaded from Scikit-learn’s Breast Cancer Wisconsin dataset.

#### Input Features (User Inputs – 5 Features):

mean radius

mean texture

mean perimeter

mean area

mean smoothness

(The model internally uses all 30 features; missing features are filled with default values.)

#### Target Variable:

0 → Malignant (High Risk of Breast Cancer)

1 → Benign (Low Risk of Breast Cancer)

### ⚙️ Technology Stack

Language: Python

Web Framework: Flask

Machine Learning: Scikit-learn

Numerical Computing: NumPy

Frontend: HTML (Jinja2 Templates)

Model: Support Vector Machine (SVM)

### 🏗️ Project Structure
project/
- │
- ├── app.py
- ├── README.md
- ├── templates/
- │   ├── index.html
- │   └── result.html
- └── static/
-   └── style.css (optional)



