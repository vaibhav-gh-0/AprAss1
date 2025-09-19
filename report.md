# Customer Retention Prediction Model

**Name:** Vaibhav Singh Tanwar
**Roll No:** 2511AI20

---

## 1. Project Overview

### Problem Statement
Predict which telecommunications customers are likely to cancel their service (churn) using machine learning algorithms.

### Dataset
- **File:** Telco Customer Churn Dataset
- **Records:**  5k+ customers
- **Features:** 20+ attributes (demographics, services, billing)

### Models Used
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (Linear)
4. Support Vector Machine (RBF)

---

## 2. Implementation

### Data Preprocessing
```python
# Load and clean data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
```

### Model Pipeline
```python
# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Example: Logistic Regression Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

# Hyperparameter tuning
search = GridSearchCV(pipeline, params, cv=5, scoring="roc_auc")
search.fit(X_train, y_train)
```

### Key Code Features
- Automated hyperparameter tuning using GridSearchCV
- Data preprocessing with scaling and encoding
- 5-fold cross-validation for model evaluation
- Multiple performance metrics (Accuracy, ROC-AUC)

---

## 3. Results and Analysis

### Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | [Insert Result] | [Insert Result] |
| K-Nearest Neighbors | [Insert Result] | [Insert Result] |
| SVM (Linear) | [Insert Result] | [Insert Result] |
| SVM (RBF) | [Insert Result] | [Insert Result] |

### Visualizations Generated

**[Screenshot: Exploratory Data Analysis]**
- Customer churn distribution
- Monthly vs Total charges analysis
- Tenure patterns

**[Screenshot: Model Performance Comparison]**
- Accuracy and ROC-AUC comparison charts
- Performance heatmap

**[Screenshot: ROC Curves]**
- ROC curve comparison for all models
<img width="1438" height="510" alt="image" src="https://github.com/user-attachments/assets/72efdeda-96cc-4a7b-bebf-906e4b71195b" />

<img width="683" height="610" alt="image" src="https://github.com/user-attachments/assets/08c67207-4593-4c79-9fb2-74bff8a77a3d" />


**[Screenshot: Confusion Matrices]**
- Individual confusion matrices showing prediction accuracy

### Key Findings
- **Best Model:** [Insert best performing model]
- **Important Features:** Contract type, tenure, monthly charges
- **Churn Rate:** [X]% of customers churn
- **Business Insight:** Month-to-month contracts have highest churn rates

### Conclusion
Successfully implemented and compared 4 machine learning models for customer churn prediction. The project demonstrates effective use of preprocessing, hyperparameter tuning, and comprehensive evaluation metrics. Results provide actionable insights for customer retention strategies.

---

**Project Files:**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

try:
    df = pd.read_csv("CustomerData.csv")
except FileNotFoundError:
    print("Error: 'CustomerData.csv'")
    print("Please make sure the dataset is in the same directory as the script.")
    exit()

# Store original data for visualization
original_df = df.copy()

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder='passthrough' 
)

log_reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))
])
log_params = {
    "pca__n_components": [10, 20, 30, None],
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l1", "l2"]
}

knn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(random_state=42)),
    ("model", KNeighborsClassifier())
])
knn_params = {
    "pca__n_components": [10, 20, 30, None],
    "model__n_neighbors": [5, 7, 9, 11],
    "model__weights": ["uniform", "distance"]
}

svc_linear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(random_state=42)),
    ("model", SVC(probability=True, random_state=42, kernel='linear'))
])
svc_linear_params = {
    "pca__n_components": [10, 20, 30, None],
    "model__C": [0.1, 1, 10]
}

svc_rbf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(random_state=42)),
    ("model", SVC(probability=True, random_state=42, kernel='rbf'))
])
svc_rbf_params = {
    "pca__n_components": [10, 20, 30, None],
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", "auto"]
}

models_to_run = {
    "Logistic Regression": (log_reg_pipeline, log_params),
    "K-Nearest Neighbors": (knn_pipeline, knn_params),
    "SVM (Linear Kernel)": (svc_linear_pipeline, svc_linear_params),
    "SVM (RBF Kernel)": (svc_rbf_pipeline, svc_rbf_params)
}

best_estimators = {}
model_results = {}

for name, (pipeline, params) in models_to_run.items():
    print(f"--- Tuning {name} ---")

    search = GridSearchCV(pipeline, params, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    best_estimators[name] = search.best_estimator_

    print(f"\nBest Parameters for {name}: {search.best_params_}")

    y_pred = search.predict(X_test)
    y_proba = search.predict_proba(X_test)[:, 1]
    
    # Store results for visualization
    model_results[name] = {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    print(f"\n{name} Performance on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50 + "\n")

# =============================================================================
# ADDITIONAL VISUALIZATIONS
# =============================================================================

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. EXPLORATORY DATA ANALYSIS PLOTS
print("Generating exploratory data analysis plots...")

# Churn Distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Exploratory Data Analysis - Customer Churn Dataset', fontsize=16, fontweight='bold')

# Churn distribution
churn_counts = original_df['Churn'].value_counts()
axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('Churn Distribution')

# Monthly charges vs Total charges colored by churn
churn_data = original_df[original_df['Churn'] == 'Yes']
no_churn_data = original_df[original_df['Churn'] == 'No']

axes[0,1].scatter(no_churn_data['MonthlyCharges'], pd.to_numeric(no_churn_data['TotalCharges'], errors='coerce'), 
                 alpha=0.6, label='No Churn', s=20)
axes[0,1].scatter(churn_data['MonthlyCharges'], pd.to_numeric(churn_data['TotalCharges'], errors='coerce'), 
                 alpha=0.6, label='Churn', s=20)
axes[0,1].set_xlabel('Monthly Charges')
axes[0,1].set_ylabel('Total Charges')
axes[0,1].set_title('Monthly vs Total Charges by Churn Status')
axes[0,1].legend()

# Tenure distribution
axes[1,0].hist([no_churn_data['tenure'], churn_data['tenure']], bins=30, alpha=0.7, 
               label=['No Churn', 'Churn'], color=['skyblue', 'salmon'])
axes[1,0].set_xlabel('Tenure (months)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Tenure Distribution by Churn Status')
axes[1,0].legend()

# Contract type vs Churn
contract_churn = pd.crosstab(original_df['Contract'], original_df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', ax=axes[1,1], rot=45)
axes[1,1].set_title('Churn Rate by Contract Type')
axes[1,1].set_ylabel('Percentage')
axes[1,1].legend(title='Churn')

plt.tight_layout()
plt.show()

# 2. MODEL PERFORMANCE COMPARISON PLOTS
print("Generating model performance comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Accuracy comparison
model_names = list(model_results.keys())
accuracies = [model_results[name]['accuracy'] for name in model_names]
roc_aucs = [model_results[name]['roc_auc'] for name in model_names]

colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
bars1 = axes[0,0].bar(model_names, accuracies, color=colors, alpha=0.8)
axes[0,0].set_title('Model Accuracy Comparison')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_ylim(0, 1)
plt.setp(axes[0,0].get_xticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# ROC-AUC comparison
bars2 = axes[0,1].bar(model_names, roc_aucs, color=colors, alpha=0.8)
axes[0,1].set_title('Model ROC-AUC Comparison')
axes[0,1].set_ylabel('ROC-AUC Score')
axes[0,1].set_ylim(0, 1)
plt.setp(axes[0,1].get_xticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, auc in zip(bars2, roc_aucs):
    height = bar.get_height()
    axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

# Model performance heatmap
performance_df = pd.DataFrame({
    'Accuracy': accuracies,
    'ROC-AUC': roc_aucs
}, index=model_names)

sns.heatmap(performance_df, annot=True, cmap='YlOrRd', ax=axes[1,0], 
            cbar_kws={'label': 'Score'}, fmt='.3f')
axes[1,0].set_title('Model Performance Heatmap')

# Combined performance plot
x_pos = np.arange(len(model_names))
width = 0.35
axes[1,1].bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8)
axes[1,1].bar(x_pos + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
axes[1,1].set_xlabel('Models')
axes[1,1].set_ylabel('Score')
axes[1,1].set_title('Accuracy vs ROC-AUC Comparison')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1,1].legend()
axes[1,1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# 3. CONFUSION MATRICES
print("Generating confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

for idx, (name, results) in enumerate(model_results.items()):
    row = idx // 2
    col = idx % 2
    
    cm = confusion_matrix(y_test, results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    axes[row, col].set_title(f'{name}')
    axes[row, col].set_xlabel('Predicted')
    axes[row, col].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 4. PRECISION-RECALL CURVES
print("Generating precision-recall curves...")

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set1(np.linspace(0, 1, len(best_estimators)))

for (name, estimator), color in zip(best_estimators.items(), colors):
    y_proba = model_results[name]['y_proba']
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax.plot(recall, precision, color=color, label=f'{name}', linewidth=2)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. ORIGINAL ROC CURVE (Enhanced)
fig, ax = plt.subplots(figsize=(10, 8))

for name, estimator in best_estimators.items():
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, name=name, ax=ax)

plt.title("ROC Curve Comparison on Test Data")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

# 6. FEATURE IMPORTANCE ANALYSIS (for models that support it)
print("Analyzing feature importance...")

# Get feature names after preprocessing
preprocessor_fitted = best_estimators['Logistic Regression'].named_steps['preprocessor']
feature_names = []

# Numerical features
feature_names.extend(num_cols.tolist())

# Categorical features (after one-hot encoding)
if hasattr(preprocessor_fitted.named_transformers_['cat'], 'get_feature_names_out'):
    cat_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names.extend(cat_feature_names.tolist())

# For Logistic Regression, plot feature coefficients
log_reg = best_estimators['Logistic Regression']
if hasattr(log_reg.named_steps['model'], 'coef_'):
    # Get coefficients
    if log_reg.named_steps.get('pca') and log_reg.named_steps['pca'].n_components_ is not None:
        print("Note: PCA was applied, so individual feature importance is not directly interpretable.")
    else:
        coef = log_reg.named_steps['model'].coef_[0]
        
        # Plot top 20 most important features
        if len(feature_names) == len(coef):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
            plt.title('Top 20 Feature Importance (Logistic Regression)')
            plt.xlabel('Absolute Coefficient Value')
            plt.tight_layout()
            plt.show()

print("\nAll visualizations have been generated successfully!")
print(f"Total models trained: {len(best_estimators)}")
print("Analysis complete!")
```
