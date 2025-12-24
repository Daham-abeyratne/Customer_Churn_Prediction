# Telecommunication Customer Churn Prediction using Machine Learning

## Overview
This project implements a machine learning solution to predict customer churn in a telecommunications service. The task is formulated as a **binary classification problem**, where the objective is to identify customers likely to discontinue their subscription so that retention strategies can be applied proactively.

The project was developed as part of the **Machine Learning** individual coursework and follows an end-to-end workflow including data exploration, preprocessing, feature selection, model training, evaluation, and ethical analysis.

---

## Dataset
- **Source:** Kaggle – Telco Customer Churn Dataset  
- **File:** `Telco-Customer-Churn.csv`  
- **Rows:** 7,043  
- **Features:** 21  
- **Target Variable:** `Churn`  
  - `1` – Customer churned  
  - `0` – Customer retained  

The dataset contains customer demographics, service usage details, contract information, and billing data.

---

## Objectives
- Perform Exploratory Data Analysis (EDA) to understand churn behavior  
- Prepare a clean and balanced dataset for modeling  
- Handle class imbalance using oversampling techniques  
- Implement and compare supervised learning models  
- Optimize performance using feature selection and threshold tuning  
- Evaluate models using business-relevant metrics  
- Discuss ethical and operational implications of deployment  

---

## Data Preprocessing & Feature Engineering
The following steps were applied:
- Converted `TotalCharges` to numeric and removed invalid entries  
- Removed non-informative identifier (`customerID`)  
- One-hot encoded categorical variables  
- Standardized numerical features using `StandardScaler`  
- Addressed class imbalance using **SMOTE** (training data only)  
- Reduced dimensionality using **SelectKBest (Mutual Information)**  
- Train–test split: **80% training / 20% testing**

---

## Models Implemented

### Decision Tree Classifier
- Interpretable, rule-based model  
- Hyperparameter tuning via grid search  
- Used as a baseline and for feature importance analysis  

### Neural Network (Final Model)
- Fully connected feedforward architecture  
- Multiple hidden layers with ReLU activation  
- Batch Normalization and Dropout for regularization  
- Adam optimizer with early stopping  
- **Decision threshold tuning** to balance Precision and Recall  

---

## Model Evaluation
Models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

### Final Neural Network Performance
- **Accuracy:** 80%  
- **Precision (Churn):** 62%  
- **Recall (Churn):** 62%  
- **F1-Score (Churn):** 0.62  
- **ROC-AUC:** 0.83  

The Neural Network was selected as the final model due to its balanced performance in identifying churners while maintaining efficient use of retention resources.

---

## Ethical & Operational Considerations
- Risk of **algorithmic bias** due to historical customer data  
- Reduced explainability compared to tree-based models  
- Recommendation to use explainability tools such as **SHAP** or **LIME**  
- Importance of monitoring data drift and model performance after deployment  
- Trade-offs between False Positives (unnecessary retention effort) and False Negatives (lost customers)

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- TensorFlow / Keras  
- Matplotlib, Seaborn  
- Google Colab  
