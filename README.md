# Credit Approval Analysis and Prediction

This project aims to analyze and predict credit approval using internal and external datasets. The analysis includes data preprocessing, feature selection using Chi-Square and ANOVA tests, and model training using XGBoost.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Selection](#feature-selection)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)

## Introduction

This project uses data from two sources: an internal bank dataset and an external dataset from a credit bureau. The goal is to merge these datasets, preprocess the data, select significant features, and train a model to predict credit approval.

## Requirements

The following Python libraries are required to run the code:

- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- statsmodels
- xgboost

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn scipy statsmodels xgboost
```

# Data Preprocessing

The data preprocessing steps include:

1. Loading the data from Excel files.
2. Replacing missing values represented by `-99999.00` with NaN.
3. Dropping rows with NaN values from the internal dataset.
4. Dropping columns with more than 10,000 NaN values and rows with NaN values from the external dataset.
5. Merging the two datasets on the `PROSPECTID` column.

# Feature Selection

## Categorical Features

- **Chi-Square Test**: Used to select significant categorical features.

## Numerical Features

- **Variance Inflation Factor (VIF)**: Used to check for multicollinearity among numerical features.
- **ANOVA Test**: Used to select significant numerical features.

The selected features are combined into a final feature set.

## Encoding

- Label encoding is applied to the `EDUCATION` column.
- One-hot encoding is applied to the `MARITALSTATUS`, `GENDER`, `last_prod_enq2`, and `first_prod_enq2` columns.

# Model Training

The dataset is split into training and testing sets. An XGBoost classifier is trained using the training set with the following parameters after Hyperparameter Tuning using Grid Search:

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

- `objective='multi:softmax'`
- `num_class=4`
- `learning_rate=0.2`
- `max_depth=3`
- `n_estimators=200`

# Evaluation

The model's performance is evaluated using accuracy, precision, recall, and F1 score. The evaluation results are printed for each class.

Accuracy: 0.78

Class p1:
Precision: 0.8466593647316539
Recall: 0.76232741617357
F1 Score: 0.8022833419823561

Class p2:
Precision: 0.8165999651992344
Recall: 0.9302279484638256
F1 Score: 0.869718309859155

Class p3:
Precision: 0.47035040431266845
Recall: 0.26339622641509436
F1 Score: 0.3376874697629415

Class p4:
Precision: 0.7398615232443125
Recall: 0.7269193391642371
F1 Score: 0.7333333333333334
