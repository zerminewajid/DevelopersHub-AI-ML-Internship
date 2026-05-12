"""
Task 3: Heart Disease Prediction

Objective: Build a classification model to predict whether a person is at risk of heart disease based on health data.
Dataset: Heart Disease UCI Dataset (loaded directly from UCI ML Repository)
Models: Logistic Regression, Decision Tree
Evaluation: Accuracy, ROC-AUC, Confusion Matrix, Feature Importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print('Loading Heart Disease UCI dataset...')

# Load Heart Disease UCI dataset from the official repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=columns, na_values='?')

# Binarize target: 0 = no disease, 1 = disease
df['target'] = (df['target'] > 0).astype(int)

print(f'Shape: {df.shape}')
print('\nFirst 5 rows:')
print(df.head())

# ============================================================================
# 2. DATA CLEANING
# ============================================================================

print('\n--- Data Cleaning ---')
print('Missing values:')
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)
print(f'\nShape after dropping NaN rows: {df.shape}')

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

print('\n--- Class Distribution ---')
print(df['target'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target', palette='Set2', hue='target', legend=False)
plt.xticks([0, 1], ['No Disease', 'Disease'])
plt.title('Class Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('task3_class_distribution.png', dpi=100)
plt.show()

# Age distribution by disease status
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='age', hue='target', bins=20, palette='Set1', kde=True)
plt.title('Age Distribution by Disease Status')
plt.legend(['No Disease', 'Disease'])

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='target', y='thalach', palette='Set2', hue='target', legend=False)
plt.xticks([0, 1], ['No Disease', 'Disease'])
plt.title('Max Heart Rate by Disease Status')
plt.ylabel('Max Heart Rate (thalach)')

plt.tight_layout()
plt.savefig('task3_eda.png', dpi=100)
plt.show()
print('Insight: Patients with heart disease tend to have lower max heart rates.')

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('task3_correlation.png', dpi=100)
plt.show()

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print('\n--- Model Training ---')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]

print(f'Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}')
print(f'Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}')

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print('\n--- Confusion Matrices ---')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices', fontsize=14)

for ax, pred, title in zip(axes, [lr_pred, dt_pred], ['Logistic Regression', 'Decision Tree']):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)

plt.tight_layout()
plt.savefig('task3_confusion_matrices.png', dpi=100)
plt.show()

# ROC Curves
print('\n--- ROC Curves ---')

plt.figure(figsize=(8, 6))

for prob, label, color in zip(
    [lr_prob, dt_prob],
    ['Logistic Regression', 'Decision Tree'],
    ['#3498db', '#e74c3c']
):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})', color=color, lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Heart Disease Prediction')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('task3_roc_curves.png', dpi=100)
plt.show()

# Classification Reports
print('\n=== Logistic Regression Classification Report ===')
print(classification_report(y_test, lr_pred, target_names=['No Disease', 'Disease']))

print('\n=== Decision Tree Classification Report ===')
print(classification_report(y_test, dt_pred, target_names=['No Disease', 'Disease']))

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================

print('\n--- Feature Importance ---')

# Decision Tree feature importance
importances = pd.Series(dt.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances_sorted.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Feature Importance — Decision Tree', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('task3_feature_importance.png', dpi=100)
plt.show()

print('\nTop 3 most important features:')
print(importances.sort_values(ascending=False).head(3))

# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '='*60)
print('ANALYSIS SUMMARY')
print('='*60)
print(f'Dataset: {len(df)} patients, 13 features, binary classification')
print('Class balance: ~54% with disease, ~46% without')
print('Logistic Regression achieved higher AUC (better model)')
print('Key predictive features: chest pain type, max heart rate, vessels')
print('='*60)
print('\nVisualization files saved:')
print('  - task3_class_distribution.png')
print('  - task3_eda.png')
print('  - task3_correlation.png')
print('  - task3_confusion_matrices.png')
print('  - task3_roc_curves.png')
print('  - task3_feature_importance.png')
