# 🫀 Heart Disease Prediction — AI & Machine Learning Project

**Prepared by:** NeuralCore Team  
* [Mohammed Hamdy Abdulrahim](https://github.com/Mohammed-Elnagar11)  
* Ahmed Yasser Abd elmaqsoud  
* Maher Sayed Abdelshahid
* Mohammed Al-Sayed Abdullah
* Mahmoud Abdelmegeed Abdelkader
* Ashraf Mohamed Abd-Elwahab

---

## 📌 Project Overview
Heart disease is one of the leading causes of death worldwide. Traditional diagnostic methods can be time-consuming and require extensive clinical tests. This project provides an AI-driven classification pipeline to predict heart disease risk at an early stage, minimizing human error and enhancing diagnostic speed and accuracy.

---

## 📊 Dataset Summary

* **Dataset Split:**
  * **Training Set:** 224 initial rows (198 after preprocessing)
  * **Testing Set:** 56 initial rows (51 after preprocessing)
* **Target Variable:** `Heart Disease` (`Yes` / `No` — Binary Classification)
* **Features (17 Columns):**
  `id`, `Age`, `Gender`, `Chest pain type`, `BP`, `Cholesterol`, `FBS over 120`, `EKG results`, `Max HR`, `Exercise angina`, `ST depression`, `Slope of ST`, `Number of vessels fluro`, `Thallium`, `work_type`, `smoking_status`, `Heart Disease`

---

## 🛠️ Data Preprocessing & Pipeline

1. **Feature Dropping:**
   * `id`: Removed (non-predictive identifier).
   * `smoking_status`: Removed after correlation analysis (contained ~20% missing/unknown values; dropping it yielded better accuracy than mode imputation).
2. **Missing Values:** Handled via listwise deletion (<4% total rows).
3. **Type Casting:** Categorical numerical codes (`Chest pain type`, `EKG results`, `Thallium`, etc.) were converted to `object` to avoid ordinal bias.
4. **Outlier Capping:** Applied Interquartile Range (IQR) capping ($Q1 - 1.5 \times IQR$ / $Q3 + 1.5 \times IQR$) derived from the training set to prevent data leakage.
5. **Encoding & Scaling:**
   * One-Hot Encoding (`pd.get_dummies()`) for categorical features.
   * Min-Max Normalization applied to numerical features.
6. **Feature Importance & Correlation:** Evaluated using **ANOVA F-test** for numerical features and **Chi-Square ($\chi^2$) test** for categorical features.

---

## 🏆 Model Performance Comparison

Multiple algorithms were trained, cross-validated, and fine-tuned using `GridSearchCV`:

| Model | Test Accuracy | Best Hyperparameters |
| :--- | :---: | :--- |
| **Logistic Regression** | **90.20%** | `{'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}` |
| **Support Vector Machine (SVM)** | **90.20%** | `{'C': 10, 'kernel': 'linear'}` |
| **K-Nearest Neighbors (KNN)** | **88.23%** | `{'n_neighbors': 11}` |
| **Random Forest** | **86.27%** | `{'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100}` |
| **Decision Tree** | **84.31%** | `{'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}` |
