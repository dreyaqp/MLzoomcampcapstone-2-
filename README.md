# MLzoomcampcapstone-2-
# Family History of Depression Prediction

## Project Overview

This project explores machine learning techniques to predict whether an individual has a **family history of depression** based on personal and demographic features. The aim is to demonstrate data preprocessing, feature engineering, and predictive modeling on structured health data.

**Task type:** Binary classification (Yes/No)

**Dataset:** Structured survey/health data (features include age, lifestyle, medical history, etc.)

---

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Modeling](#modeling)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Folder Structure](#folder-structure)


---

## Project Motivation

Depression is a significant mental health issue worldwide. Research shows that family history is a strong risk factor. Predictive models can help identify individuals at higher risk, guiding early intervention and preventive measures.

---

## Dataset

* The dataset contains features such as demographics, lifestyle choices, and medical history.
* Target variable: `family_history_of_depression` (`Yes` or `No`)
* Sources: [https://www.kaggle.com/datasets/anthonytherrien/depression-dataset]

---

## Preprocessing

1. **Target Encoding:**

   * Converted `Yes` → `1` and `No` → `0` for regression/classification compatibility.



2. **Scaling:**

   * Standardized numeric features using `StandardScaler`.

---

## Modeling

* Baseline model: **Linear Regression**
* Pipeline:

  ```python
  Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler()),
      ('lr', LinearRegression())
  ])
  ```
* Optional: Classification models like `LogisticRegression` or `RandomForestClassifier` can also be used for better interpretability.

---

## Evaluation Metrics

* **Regression metrics** (if using numeric target):

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)
  * R² Score


## Quickstart (minimal)
Use Docker (recommended, ensures correct Python and deps):

docker build -t depression-api:latest .
# Run the container
docker run -p 8000:8000 depression-api:latest


```


