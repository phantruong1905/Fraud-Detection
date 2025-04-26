# Credit Card Fraud Detection using LSTM_Attention and Ensemble Learning Models

## Overview

This project focuses on detecting fraudulent transactions in a credit card dataset using both traditional ensemble learning models and an improved deep learning approach: LSTM networks with an attention mechanism. The goal is to classify fraudulent and non-fraudulent transactions effectively, minimizing undetected frauds.

## Dataset

The dataset used is the [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) from Kaggle. It is highly imbalanced, with fraudulent transactions representing only about 0.57% of the total data.

## Models

- **Traditional ML Models**:
  - Random Forest
  - Logistic Regression
  - XGBoost
  - AdaBoost

- **Deep Learning Model**:
  - **LSTM with Attention Layer**:
    - LSTM layers to capture sequential dependencies in transaction patterns.
    - Attention layer on top of LSTM outputs to focus on critical timesteps.
    - **No need resampling techniques** like SMOTE used.

## Results

| Model               | Train F1-Score | Test F1-Score | ROC Area | Precision | Recall  | F1-Score | Support  |
|---------------------|----------------|---------------|----------|-----------|---------|----------|----------|
| XGBoost             | 0.996297       | 0.651437      | 0.933808 | 0.519796  | 0.872368| 0.651437 | 259335   |
| AdaBoost            | 0.987126       | 0.488228      | 0.921481 | 0.342043  | 0.852632| 0.488228 | 259335   |
| Cost-Sensitive RF   | 1.000000       | 0.719955      | 0.922191 | 0.625850  | 0.847368| 0.719955 | 259335   |
| Bagging             | 0.999989       | 0.679446      | 0.917875 | 0.570662  | 0.839474| 0.679446 | 259335   |
| Random Forest       | 1.000000       | 0.719886      | 0.915355 | 0.633500  | 0.833553| 0.719886 | 259335   |
| Logistic Regression | 0.840756       | 0.139310      | 0.860843 | 0.076514  | 0.776974| 0.139310 | 259335   |
| **LSTM + Attention** | 0.901484         | 0.902376        | 0.911245	|0.895127	|0.910567	|0.902376| 259335   |
