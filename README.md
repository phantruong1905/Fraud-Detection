# Credit Card Fraud Detection using Ensemble Learning Models

## Overview

This project focuses on detecting fraudulent transactions in a credit card dataset using several ensemble learning techniques. The dataset used is the [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) available on Kaggle. The goal is to build and evaluate models to classify fraudulent and non-fraudulent transactions.

## Dataset

The dataset contains a mix of genuine and fraudulent credit card transactions, including features such as transaction amount, customer demographic data, transaction location, and more. The data is highly imbalanced, with far fewer fraud cases (0.57%) compared to non-fraudulent ones (99.43%).

### Dataset Source

[Kaggle - Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)

### Data Preprocessing

- **Resampling Techniques**: To handle the class imbalance.
- **Feature Importance**: Based on feature importance from Random Forest, key features like transaction amount (`amt`) and age were found to be significant.
- **Feature Selection**: Filtering was performed based on importance to retrain models with significant features only.

## Models Used

Several ensemble learning methods were employed to detect fraudulent transactions. Below is a list of models used and the metrics evaluated:

1. **XGBoost** (eXtreme Gradient Boosting)
2. **AdaBoost** (Adaptive Boosting)
3. **Cost-Sensitive Random Forest**
4. **Bagging**
5. **Random Forest** (Standard)
6. **Logistic Regression**

## Evaluation Metrics

The following metrics were used to evaluate the models:

- **Train F1-Score**: The F1 score on the training set.
- **Test F1-Score**: The F1 score on the test set.
- **Overfitting**: Whether the model shows overfitting behavior.
- **ROC Area (AUC)**: Area under the ROC curve.
- **Precision**: The proportion of true positive predictions in the positive class.
- **Recall**: The proportion of true positives identified in the actual positive class (very important for fraud detection).
- **Support**: The number of test samples.

## Results

The table below summarizes the results of each model:

| Model               | Train F1-Score | Test F1-Score | Overfitting | ROC Area | Precision | Recall  | F1-score | Support  |
|---------------------|----------------|---------------|-------------|----------|-----------|---------|----------|----------|
| XGBoost             | 0.996297       | 0.651437      | True        | 0.933808 | 0.519796  | 0.872368| 0.651437 | 259335   |
| AdaBoost            | 0.987126       | 0.488228      | True        | 0.921481 | 0.342043  | 0.852632| 0.488228 | 259335   |
| Cost-Sensitive RF   | 1.000000       | 0.719955      | True        | 0.922191 | 0.625850  | 0.847368| 0.719955 | 259335   |
| Bagging             | 0.999989       | 0.679446      | True        | 0.917875 | 0.570662  | 0.839474| 0.679446 | 259335   |
| Random Forest       | 1.000000       | 0.719886      | True        | 0.915355 | 0.633500  | 0.833553| 0.719886 | 259335   |
| Logistic Regression | 0.840756       | 0.139310      | True        | 0.860843 | 0.076514  | 0.776974| 0.139310 | 259335   |

### Key Findings

- **XGBoost**: Performed well with a high ROC AUC (0.9338) and recall (0.872), but suffered from overfitting with a significant drop in F1-score between training and testing. However, this model has very low training time compared to the others.
  
- **Cost-Sensitive Random Forest**: Achieved the highest test F1-score (0.7199) and maintained a strong balance between precision (0.6258) and recall (0.8473). The model effectively handles class imbalance, making it a strong candidate for fraud detection.

- **AdaBoost**: Showed moderate performance with a decent recall (0.8526) but relatively low precision (0.3420), leading to a lower F1-score (0.4882).

- **Bagging** and **Random Forest**: Both achieved similar recall values (0.8394 for Bagging, 0.8336 for RF) but Bagging had a slightly lower F1-score due to lower precision.

- **Logistic Regression**: Performed poorly across all metrics, with an F1-score of 0.139 on the test set, showing its limitation in handling the imbalanced nature of the dataset.

### Emphasis on Recall for Fraud Detection

In fraud detection tasks, recall is a critical metric since it measures the model's ability to correctly identify fraudulent transactions. High recall ensures that most frauds are caught, even at the expense of lower precision (which means more false positives). In this context, **XGBoost** and **Cost-Sensitive Random Forest** performed well, with recall values of 0.872 and 0.847 respectively, making them effective for this purpose.

## Conclusion

The project demonstrates that ensemble learning methods such as XGBoost and Cost-Sensitive Random Forest can handle imbalanced datasets effectively, with high recall values crucial for detecting fraud. While XGBoost excelled in terms of speed, **Cost-Sensitive Random Forest** showed the best overall balance between precision and recall, making it the most suitable model for this task.

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset** from Kaggle:
   ```bash
   !kaggle datasets download -d priyamchoksi/credit-card-transactions-dataset
   ```

3. **Run the notebook**:
   Load the Jupyter notebook and run all cells to preprocess the data, train the models, and evaluate the results.

## Future Work

- Explore more advanced feature engineering techniques.
- Address overfitting by tuning hyperparameters or using cross-validation strategies.
