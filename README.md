# APR Ass1

# Customer Retention Prediction Model

A machine learning project that predicts customer churn using classification algorithms.

## Project Description

This project analyzes telecom customer data to predict which customers are likely to cancel their service. The program compares four different machine learning models and provides detailed performance analysis.

## Dataset

- **File**: `CustomerData.csv`
- **Records**: Customer information including demographics, services, and billing data
- **Target**: Churn status (Yes/No)

## Models Used

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (Linear)
4. Support Vector Machine (RBF)

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

1. Place the CSV dataset in the project folder
2. Run the Python script:
   ```bash
   python churn_prediction.py
   ```

## Output

- Model accuracy and ROC-AUC scores
- Performance comparison charts
- Confusion matrices
- ROC curves
- Feature importance analysis

## Key Features

- Automated hyperparameter tuning
- Data preprocessing with scaling and encoding
- Multiple visualization charts
- Model performance comparison


