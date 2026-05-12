"""
Task 1: Exploring and Visualizing the Iris Dataset

Objective: Load, inspect, and visualize the Iris dataset to understand data trends and distributions.
Dataset: Iris Dataset (loaded via seaborn)
Models/Tools: pandas, matplotlib, seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# 1. LOAD AND INSPECT DATA
# ============================================================================

# Load Iris dataset
df = sns.load_dataset('iris')
print('Dataset loaded successfully.')

# Basic Inspection
print('\nShape:', df.shape)
print('\nColumns:', df.columns.tolist())
print('\nFirst 5 rows:')
print(df.head())

print('\n--- Dataset Info ---')
df.info()

print('\n--- Descriptive Statistics ---')
print(df.describe())

print('\n--- Missing Values ---')
print('Missing values per column:')
print(df.isnull().sum())

print('\n--- Class Distribution ---')
print(df['species'].value_counts())

# ============================================================================
# 2. SCATTER PLOT - FEATURE RELATIONSHIPS
# ============================================================================

# Pairplot shows scatter relationships between all feature pairs, colored by species
sns.pairplot(df, hue='species', palette='Set2', diag_kind='kde')
plt.suptitle('Iris Dataset — Pairplot of Features by Species', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('task1_pairplot.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. HISTOGRAMS - FEATURE DISTRIBUTIONS
# ============================================================================

# Create histograms for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    for species in df['species'].unique():
        data = df[df['species'] == species][feature]
        ax.hist(data, label=species, alpha=0.6, bins=15)
    
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('task1_histograms.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. CORRELATION HEATMAP
# ============================================================================

# Calculate correlation matrix
correlation_matrix = df.iloc[:, :-1].corr()

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Iris Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_correlation.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. BOX PLOTS - OUTLIERS AND DISTRIBUTION BY SPECIES
# ============================================================================

# Box plots for each feature grouped by species
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    sns.boxplot(data=df, x='species', y=feature, ax=ax, palette='Set2')
    ax.set_title(f'{feature} by Species', fontsize=12, fontweight='bold')
    ax.set_xlabel('Species', fontsize=11)
    ax.set_ylabel(feature, fontsize=11)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('task1_boxplots.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================

print('\n' + '='*60)
print('SUMMARY STATISTICS BY SPECIES')
print('='*60)

for species in df['species'].unique():
    print(f'\n{species.upper()}:')
    species_data = df[df['species'] == species].iloc[:, :-1]
    print(species_data.describe())

print('\n' + '='*60)
print('ANALYSIS COMPLETE')
print('='*60)
print('\nVisualization files saved:')
print('  - task1_pairplot.png')
print('  - task1_histograms.png')
print('  - task1_correlation.png')
print('  - task1_boxplots.png')
