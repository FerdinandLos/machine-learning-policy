import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import os
import numpy as np

# Load the urban emissions panel dataset
csv_path = Path(__file__).resolve().parents[1] / "Data" / "urban_emissions_panel.csv"
df = pd.read_csv(csv_path)

print(df.head())

# Ensure the figures directory exists
save_dir = os.path.join('Writing', 'Figures')
os.makedirs(save_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Check for Missing Values (NaNs)
# ---------------------------------------------------------
print("--- MISSING VALUES SUMMARY ---")
missing_data = df.isna().sum()
missing_cols = missing_data[missing_data > 0]

if missing_cols.empty:
    print("Good news: No missing values found in the dataset.\n")
else:
    print("Missing values found in the following columns:")
    print(missing_cols.to_string())
    print("\n")

# ---------------------------------------------------------
# 2. Check for Outliers and Data Errors
# ---------------------------------------------------------
print("--- DESCRIPTIVE STATISTICS (Outlier Detection) ---")
# We select a few key continuous variables to inspect
check_cols = ['transport_co2', 'total_co2', 'population', 'area_km2', 'gdp_pc']

# df.describe() gives us count, mean, std, min, 25%, 50%, 75%, and max
summary_stats = df[check_cols].describe().round(2)
print(summary_stats)
print("\n* Tip: Compare the 'min' and 'max' against the 'mean'. Look for negative populations, zero emissions, or implausibly massive jumps.\n")

# ---------------------------------------------------------
# 3. Assess the Distribution for Log Transformation
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Raw Distribution
ax1.hist(df['transport_co2'].dropna(), bins=40, color='gray', edgecolor='black')
ax1.set_title('Raw Distribution of Transport $CO_2$')
ax1.set_xlabel('Transport $CO_2$ (kt)')
ax1.set_ylabel('Frequency')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Log-Transformed Distribution
# We use np.log1p (log(1+x)) to safely handle any potential exact zeros
ax2.hist(np.log1p(df['transport_co2'].dropna()), bins=40, color='black', edgecolor='white')
ax2.set_title('Log-Transformed Distribution')
ax2.set_xlabel('Log(Transport $CO_2$)')
ax2.set_ylabel('Frequency')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'co2_distribution.pdf'), format='pdf', bbox_inches='tight')

#---------------------------------------------------------
# 4. Check all variables for skewed distributions
#---------------------------------------------------------

# 1. Define variables that should NEVER be log-transformed
# This includes IDs, years, binary dummies, and categorical/ordinal variables
exclude_cols = [
    'city_id', 'year', 'country_id', 'cluster_id', 'latitude_zone',
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year',
    'cp_announce_year', 'lez_announce_year', 'national_climate_pact', 'political_green'
]

# Select only numeric columns that are not in the exclude list
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

# 2. Calculate Skewness
# We use df.skew() which calculates the sample skewness
skewness = df[numeric_cols].skew()

# Filter for highly skewed variables (absolute value > 1)
# Sort them so the most severely skewed variables are at the top
highly_skewed_cols = skewness[abs(skewness) > 1].sort_values(ascending=False)

print("--- HIGHLY SKEWED VARIABLES (Skewness > 1 or < -1) ---")
if highly_skewed_cols.empty:
    print("No highly skewed variables detected.")
else:
    print(highly_skewed_cols)
    print("\n* Note: Positive values indicate a right-skew (long tail to the right).")

# 3. Visual Confirmation (Automated Grid Plot)
if not highly_skewed_cols.empty:
    cols_to_plot = highly_skewed_cols.index.tolist()
    
    # Dynamically calculate the grid size (3 columns wide)
    n_cols = 3
    n_rows = math.ceil(len(cols_to_plot) / n_cols)
    
    # Academic formatting
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    
    # Flatten axes array for easy iteration, handling cases where n_rows=1
    if n_rows > 1:
        axes = axes.flatten()
    elif len(cols_to_plot) > 1:
        axes = axes
    else:
        axes = [axes] 
        
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(df[col].dropna(), bins=30, color='gray', edgecolor='black')
        axes[i].set_title(f'{col}\n(Skew: {highly_skewed_cols[col]:.2f})')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_yticks([]) # Hide y-ticks to keep the grid clean
        
    # Hide any unused subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    
    # Save the diagnostic plot
    save_dir = os.path.join('Writing', 'Figures')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'skewness_diagnostics.pdf'), format='pdf', bbox_inches='tight')
    
    plt.show()