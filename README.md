# Fraud Detection in Financial Transactions

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Descriptions](#model-descriptions)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project focuses on detecting fraud in financial transactions using various machine learning models. The models implemented range from linear models like Logistic Regression to ensemble methods like Random Forest and XGBoost, as well as anomaly detection techniques like Isolation Forest and Local Outlier Factor (LOF). The goal is to identify abnormal behaviors in financial transactions accurately and robustly.

## Features
- Data preprocessing including normalization and oversampling.
- Implementation of multiple machine learning models:
  - Logistic Regression(LR)
  - Support Vector Machine (SVM)
  - Random Forest(RF)
  - XGBoost
  - Isolation Forest(IF)
  - Local Outlier Factor (LOF)
  - Recurrent Neural Network (RNN)
- Hyperparameter tuning using Grid Search for optimized model performance.
- Evaluation metrics including confusion matrix and classification report.

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
   https://github.com/Lemoninmountain/Enhancing-Fraud-Detection-in-the-Ethereum-Blockchain-Using-Ensemble-Stacking-Machine-Learning.git
2. Installation Environment
   Download the requirements.yml file into your defult dictionary
   Open terminal
   cd "your environment path"
   conda env create -f requirements.yml
3. Download .ipynb file run it.
4. Download the transaction_dataset.csv and place it in the directory.

##Usage
1. Use the env environment to run the code

##Model Descriptions
1. Logistic Regression(LR)
A linear model used for binary classification. It maps the linear combination of features into a probability range for classification.

2. Support Vector Machine (SVM)
A nonlinear model that finds the optimal hyperplane in the feature space for classification.

3. Random Forest(RF)
An ensemble learning method composed of multiple decision trees, which improves accuracy and robustness by voting or averaging.

4. XGBoost
A gradient boosting framework that iteratively trains decision trees and combines them to enhance prediction performance. Hyperparameters are tuned using Grid Search.

5. Isolation Forest(IF)
An anomaly detection method that identifies outliers by constructing trees through random feature and split selection.

6. Local Outlier Factor (LOF)
An anomaly detection method that detects outliers by comparing the local density of a data point to that of its neighbors.

7. Recurrent Neural Network (RNN)
A neural network model suitable for sequential data, capturing temporal dependencies.

##Evaluation
Evaluation metrics used include:

  ·Confusion Matrix
  ·Classification Report (Precision, Recall, F1-Score)

##Results
The proposed methods show high accuracy and robustness in identifying abnormal behaviors in financial transactions. Detailed evaluation results and model performance are provided in the .ipynb.
