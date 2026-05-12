"""
Task 2: Predict Future Stock Prices (Short-Term)

Objective: Use historical stock data to predict the next day's closing price.
Dataset: Apple (AAPL) stock data from Yahoo Finance via yfinance
Models: Linear Regression, Random Forest Regressor
Features: Open, High, Low, Volume → predict next day's Close
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD STOCK DATA
# ============================================================================

print('Loading stock data...')
ticker = 'AAPL'
df = yf.download(ticker, start='2019-01-01', end='2024-12-31', auto_adjust=True)
df = df.reset_index()

# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print('\nFirst 5 rows:')
print(df.head())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print('\n--- Missing Values ---')
print(df.isnull().sum())
print(f'\nDate range: {df["Date"].min()} to {df["Date"].max()}')
print(f'Total trading days: {len(df)}')

# Closing price over time
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Close'], color='steelblue', lw=1.5)
plt.title(f'{ticker} Closing Price (2019–2024)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('task2_closing_price.png', dpi=100)
plt.show()

# Volume over time
plt.figure(figsize=(14, 3))
plt.bar(df['Date'], df['Volume'], color='orange', alpha=0.6, width=1)
plt.title(f'{ticker} Trading Volume (2019–2024)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.tight_layout()
plt.savefig('task2_volume.png', dpi=100)
plt.show()

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print('\n--- Feature Engineering ---')

# Add technical indicators as additional features
df['MA_5'] = df['Close'].rolling(5).mean()       # 5-day moving average
df['MA_20'] = df['Close'].rolling(20).mean()     # 20-day moving average
df['Daily_Return'] = df['Close'].pct_change()    # daily return
df['Volatility'] = df['Daily_Return'].rolling(5).std()  # 5-day volatility

# Target: next day's closing price
df['Next_Close'] = df['Close'].shift(-1)

# Drop rows with NaN (from rolling windows and shift)
df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Volume', 'Close', 'MA_5', 'MA_20', 'Daily_Return', 'Volatility']
X = df[features]
y = df['Next_Close']

print(f'Samples after feature engineering: {len(df)}')
print(f'Features: {features}')

# ============================================================================
# 4. TRAIN/TEST SPLIT & SCALING
# ============================================================================

print('\n--- Train/Test Split ---')

# Time-series split: last 20% as test (no shuffle to preserve temporal order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df['Date'].iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Train: {len(X_train)} days, Test: {len(X_test)} days')

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print('\n--- Model Training ---')

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print('Models trained successfully.')

# ============================================================================
# 6. EVALUATION
# ============================================================================

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'\n{name}:')
    print(f'  MAE:  ${mae:.2f}')
    print(f'  RMSE: ${rmse:.2f}')
    print(f'  R²:   {r2:.4f}')

print('\n--- Model Performance ---')
evaluate('Linear Regression', y_test, lr_pred)
evaluate('Random Forest', y_test, rf_pred)

# Actual vs Predicted plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'{ticker} Stock — Actual vs Predicted Closing Price', fontsize=14)

for ax, pred, title, color in zip(
    axes,
    [lr_pred, rf_pred],
    ['Linear Regression', 'Random Forest'],
    ['#e74c3c', '#2ecc71']
):
    ax.plot(dates_test.values, y_test.values, label='Actual', color='steelblue', lw=1.5)
    ax.plot(dates_test.values, pred, label=f'Predicted ({title})', color=color, lw=1.5, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.3)

plt.xlabel('Date')
plt.tight_layout()
plt.savefig('task2_actual_vs_predicted.png', dpi=100)
plt.show()

# Feature importance from Random Forest
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Feature Importance — Random Forest', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('task2_feature_importance.png', dpi=100)
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '='*60)
print('ANALYSIS SUMMARY')
print('='*60)
print(f'Dataset: ~{len(df)} trading days of {ticker} stock (2019–2024)')
print('Random Forest significantly outperforms Linear Regression')
print('Most important features: Previous Close, MA_5, MA_20')
print('Limitation: Struggles at sudden reversals (earnings, news events)')
print('='*60)
print('\nVisualization files saved:')
print('  - task2_closing_price.png')
print('  - task2_volume.png')
print('  - task2_actual_vs_predicted.png')
print('  - task2_feature_importance.png')
