"""
Task 6: House Price Prediction (Regression)

Objective: Predict house prices using property features such as size, bedrooms, and location.
Dataset: California Housing Dataset (from sklearn — real-world housing data)
Models: Linear Regression, Gradient Boosting Regressor
Evaluation: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print('Loading California Housing dataset...')

# California Housing dataset: 20,640 samples, 8 features
# Target: median house value (in $100,000 units)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Convert target to actual dollar values for interpretability
df['MedHouseVal'] = df['MedHouseVal'] * 100_000

print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print('\nFirst 5 rows:')
print(df.head())

# ============================================================================
# 2. DATA INSPECTION & CLEANING
# ============================================================================

print('\n--- Data Inspection ---')
print('Missing values:')
print(df.isnull().sum())

print('\nDescriptive Statistics:')
print(df.describe())

# Feature descriptions
feature_desc = {
    'MedInc': 'Median income (in $10,000)',
    'HouseAge': 'Median house age (years)',
    'AveRooms': 'Average rooms per household',
    'AveBedrms': 'Average bedrooms per household',
    'Population': 'Block population',
    'AveOccup': 'Average household occupancy',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'MedHouseVal': 'Median house value (target, $)'
}
print('\nFeature descriptions:')
for k, v in feature_desc.items():
    print(f'  {k}: {v}')

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print('\n--- Exploratory Data Analysis ---')

# Price distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df['MedHouseVal'], bins=50, color='steelblue', edgecolor='white')
plt.title('House Price Distribution')
plt.xlabel('Median House Value ($)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df['MedHouseVal']), bins=50, color='#e74c3c', edgecolor='white')
plt.title('Log-Transformed Price Distribution')
plt.xlabel('log(Median House Value)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('task6_price_distribution.png', dpi=100)
plt.show()
print('Insight: Prices are right-skewed; there is a hard cap at $500,000 in the original data.')

# Geographic price distribution
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df['Longitude'], df['Latitude'],
    c=df['MedHouseVal'], cmap='RdYlGn',
    alpha=0.4, s=5
)
plt.colorbar(scatter, label='Median House Value ($)')
plt.title('California House Prices by Location', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('task6_geo_prices.png', dpi=100)
plt.show()
print('Insight: Coastal areas (SF Bay Area, LA) have significantly higher prices.')

# Correlation with target
correlations = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values()

plt.figure(figsize=(8, 5))
correlations.plot(kind='barh',
                  color=['#e74c3c' if x < 0 else '#2ecc71' for x in correlations],
                  edgecolor='white')
plt.title('Feature Correlation with House Price', fontsize=14)
plt.xlabel('Pearson Correlation')
plt.axvline(0, color='black', lw=0.8)
plt.tight_layout()
plt.savefig('task6_correlations.png', dpi=100)
plt.show()
print('Insight: MedInc (median income) is the strongest predictor of house prices.')

# ============================================================================
# 4. FEATURE ENGINEERING & PREPROCESSING
# ============================================================================

print('\n--- Feature Engineering & Preprocessing ---')

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Train samples: {len(X_train)}, Test samples: {len(X_test)}')

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print('\n--- Model Training ---')

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

print('Models trained successfully.')

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

print('\n--- Model Evaluation ---')

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'\n{name}:')
    print(f'  MAE:  ${mae:,.0f}')
    print(f'  RMSE: ${rmse:,.0f}')
    print(f'  R²:   {r2:.4f}')

evaluate('Linear Regression', y_test, lr_pred)
evaluate('Gradient Boosting', y_test, gb_pred)

# Actual vs Predicted plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Actual vs Predicted House Prices', fontsize=14)

sample_idx = np.random.choice(len(y_test), 500, replace=False)
y_test_arr = np.array(y_test)

for ax, pred, title, color in zip(
    axes,
    [lr_pred, gb_pred],
    ['Linear Regression', 'Gradient Boosting'],
    ['#3498db', '#e74c3c']
):
    ax.scatter(y_test_arr[sample_idx] / 1000, pred[sample_idx] / 1000,
               alpha=0.4, s=15, color=color)
    lims = [0, 550]
    ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual Price ($1000s)')
    ax.set_ylabel('Predicted Price ($1000s)')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig('task6_actual_vs_predicted.png', dpi=100)
plt.show()

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print('\n--- Feature Importance Analysis ---')

# Feature importance from Gradient Boosting
importances = pd.Series(gb.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances_sorted.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Feature Importance — Gradient Boosting', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('task6_feature_importance.png', dpi=100)
plt.show()

print('\nTop 3 most important features:')
print(importances.sort_values(ascending=False).head(3))

# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '='*70)
print('ANALYSIS SUMMARY')
print('='*70)
print(f"""
Dataset:
  - Samples: {len(df):,}
  - Features: 8
  - Target: Median house value ($)
  
Key Findings:
  - Gradient Boosting significantly outperforms Linear Regression
  - Top predictors: Median Income, Latitude, Longitude
  - Geographic insight: Coastal areas have highest prices
  - Non-linear relationships between features and price
  
Model Comparison:
  - Linear Regression: assumes linear relationships
  - Gradient Boosting: captures complex, non-linear patterns
  
Limitations:
  - Data has a hard cap at $500,000 (artificially censored)
  - Temporal factors not considered (housing market changes)
  - Limited to California data
""")
print('='*70)
print('\nVisualization files saved:')
print('  - task6_price_distribution.png')
print('  - task6_geo_prices.png')
print('  - task6_correlations.png')
print('  - task6_actual_vs_predicted.png')
print('  - task6_feature_importance.png')
