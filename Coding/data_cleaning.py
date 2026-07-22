import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
check_cols = ['transport_co2', 'total_co2', 'population', 'area_km2', 'gdp_pc']
summary_stats = df[check_cols].describe().round(2)
print(summary_stats)
print("\n* Tip: Compare the 'min' and 'max' against the 'mean'. Look for negative populations, zero emissions, or implausibly massive jumps.\n")

# ---------------------------------------------------------
# 3. Assess the Distribution for Log Transformation
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(df['transport_co2'].dropna(), bins=40, color='gray', edgecolor='black')
ax1.set_title('Raw Distribution of Transport $CO_2$')
ax1.set_xlabel('Transport $CO_2$ (kt)')
ax1.set_ylabel('Frequency')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

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
exclude_cols = [
    'city_id', 'year', 'country_id', 'cluster_id', 'latitude_zone',
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year',
    'cp_announce_year', 'lez_announce_year', 'national_climate_pact', 
    'coastal', 'political_green',
    'unemployment', 'education_share', 'renewable_electricity_share', 
    'fleet_diesel_share', 'fleet_petrol_share', 'fleet_electric_share', 
    'industry_manufacturing', 'industry_services', 'industry_logistics', 
    'industry_public',
    'museum_visitors_pc', 'library_count', 'streetlight_density', 
    'fountain_count', 'bench_count_pc', 'flagpole_count', 'sister_city_count',
    'public_transit_score', 'logistics_activity', 'fiscal_capacity', 
    'electoral_competitiveness', 'ngo_environment_index'
]

numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
skewness = df[numeric_cols].skew()
highly_skewed_cols = skewness[abs(skewness) > 1].sort_values(ascending=False)

print("--- HIGHLY SKEWED VARIABLES (Skewness > 1 or < -1) ---")
if highly_skewed_cols.empty:
    print("No highly skewed variables detected.")
else:
    print(highly_skewed_cols)
    print("\n* Note: Positive values indicate a right-skew (long tail to the right).")

if not highly_skewed_cols.empty:
    cols_to_plot = highly_skewed_cols.index.tolist()
    n_cols = 3
    n_rows = math.ceil(len(cols_to_plot) / n_cols)
    
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    
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
        axes[i].set_yticks([])
        
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'skewness_diagnostics.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# 5. Apply Log Transformations
# ---------------------------------------------------------
covariates_to_log = [
    'total_co2', 'population', 'pop_density',
    'gdp_pc', 'area_km2', 'electricity_price', 'fuel_price'
]

df['log_transport_co2'] = np.log(df['transport_co2'])

for col in covariates_to_log:
    df[f'log_{col}'] = np.log(df[col])

# Drop the raw unlogged columns
columns_to_drop = ['transport_co2'] + covariates_to_log
df = df.drop(columns=columns_to_drop)

# --------------------------
# 6. Find the optimal K clusters
# --------------------------

# 1. Isolate exogenous variables and account for panel structure
# Define columns that must NOT be used for clustering
# UPDATED: Swapped raw co2 columns for their log counterparts
K_means_exclude_cols = ['year', 'log_transport_co2', 'log_total_co2', 'cp_active', 'lez_active', 
                'cp_impl_year', 'lez_impl_year', 'cp_announce_year', 'lez_announce_year', 'country_id']

# Drop excluded columns and aggregate to one row per city (mean over time)
city_features = df.drop(columns=K_means_exclude_cols).groupby('city_id').mean()

# 2. Standardize the data (crucial for distance-based K-Means)
X_scaled = StandardScaler().fit_transform(city_features)

# 3. Efficiently compute both metrics in a single loop
k_range = range(2, 10)
inertias, silhouettes = [], []

for k in k_range:
    # n_init='auto' optimizes initialization to save computation time
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# 4. Academic Figure Formatting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot 1: Elbow Method
ax1.plot(k_range, inertias, marker='s', color='black', linestyle='-', linewidth=1.5, markersize=5)
ax1.set(title='Elbow Method (Inertia)', xlabel='Number of Clusters (K)', ylabel='Inertia')
ax1.grid(True, linestyle='--', alpha=0.3) # Faint grid for readability
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Silhouette Analysis
ax2.plot(k_range, silhouettes, marker='o', color='black', linestyle='-', linewidth=1.5, markersize=5)
ax2.set(title='Silhouette Analysis', xlabel='Number of Clusters (K)', ylabel='Silhouette Score')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

# 4. Save to Specific Directory
# os.makedirs ensures the code doesn't crash if the folders don't exist yet
save_dir = os.path.join('Writing', 'Figures')
os.makedirs(save_dir, exist_ok=True)

# Save as both PNG (for quick viewing) and PDF (ideal for LaTeX/Word documents)
pdf_path = os.path.join(save_dir, 'kmeans_evaluation.pdf')

plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

# 5. Apply the optimal K = 2 following Silhouette Analysis
optimal_k = 2
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(X_scaled)

# Assign the fixed cluster IDs back to the aggregated dataframe
city_features['cluster_id'] = final_kmeans.labels_

# Merge the fixed cluster IDs back to the original panel dataset based on city_id
df = df.merge(city_features[['cluster_id']], on='city_id', how='left')

# ----------
# Show how the two types differ
# ----------
# 1. Load the Variable Descriptions for Real Names
# Assuming the file is in a folder named 'Data' relative to your script
desc_path = os.path.join('Data', 'variable_descriptions.csv')
descriptions_df = pd.read_csv(desc_path)

# Create a dictionary mapping 'variable_name' to 'description'
name_mapping = dict(zip(descriptions_df['variable_name'], descriptions_df['description']))


# 2. Calculate raw means for interpretability
cluster_means = city_features.groupby('cluster_id').mean()

# Identify the top 10 most distinguishing features
percent_diff = abs(cluster_means.loc[0] - cluster_means.loc[1]) / cluster_means.mean()
top_features = percent_diff.sort_values(ascending=False).head(10).index

# Change the names to be less long
# Create a dictionary for the specific variables that are too long
custom_short_names = {
    "Indicator: city participates in national climate pact (1=yes; persistent from 2016 once joined)": "National Climate Pact Participation",
    "Temperature anomaly relative to 1991-2020 baseline (C)": "Temperature Anomaly",
    "Green party vote share or equivalent index (0-1)": "Green Party Vote Share",
    "Latitude zone (1=northern, 2=central, 3=southern)": "Latitude Zone",
    "Environmental NGO activity index (0-100)": "Environmental NGO Activity Index",
    "log_pop_density": "Log Population per $km^2$",
    # Add any others that need shortening here
}


# 3. Create a Summary Table for the Main Text
summary_table = cluster_means[top_features].T

# Rename columns to your chosen typology names
summary_table.columns = ['Sprawling Regional Hubs', 'Dense Progressive Metropolises']

# Map the raw variable names in the index to the real descriptions
summary_table = summary_table.rename(index=name_mapping)

# Apply custom short names
summary_table = summary_table.rename(index=custom_short_names)

# Enforce two-digit rounding
summary_table = summary_table.round(2)

print("Top 10 Distinguishing Features:")
print(summary_table)

# Optional exports
summary_table.to_latex('Writing/Tables/cluster_summary.tex', float_format="%.2f")
# summary_table.to_csv('Writing/Tables/cluster_summary.csv')


# 4. Create an Academic Visualization (Standardized Differences)
X_scaled_df = pd.DataFrame(X_scaled, columns=city_features.drop(columns='cluster_id').columns)
X_scaled_df['cluster_id'] = city_features['cluster_id'].values
scaled_means = X_scaled_df.groupby('cluster_id').mean()

# Calculate the difference in standard deviations
scaled_diff = (scaled_means.loc[1] - scaled_means.loc[0])[top_features]

# Map the raw variable names to the real descriptions for the plot's y-axis
scaled_diff = scaled_diff.rename(index=name_mapping)

# Apply custom short names for the plot
scaled_diff = scaled_diff.rename(index=custom_short_names)

fig, ax = plt.subplots(figsize=(8, 5))

# Create a horizontal bar chart
# Positive values mean Type 1 is higher; negative means Type 0 is higher
colors = ['black' if val > 0 else 'gray' for val in scaled_diff]
scaled_diff.plot(kind='barh', color=colors, ax=ax)

ax.set_title('Top 10 Distinguishing City Characteristics')
ax.set_xlabel('Difference in Standard Deviations\n(Dense Metropolises - Sprawling Hubs)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Invert y-axis so the biggest difference is at the top
plt.gca().invert_yaxis() 
plt.tight_layout()

# Save the figure
save_dir = os.path.join('Writing', 'Figures')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'cluster_differences.pdf'), format='pdf', bbox_inches='tight')


# ---------------------------------------------------------
# 7. Construct Panel Heterogeneity Proxies (Textbook Method)
# ---------------------------------------------------------
print("\n--- CONSTRUCTING PANEL HETEROGENEITY PROXIES ---")

# Define the dynamic covariates (X_{i,t}) by excluding IDs, static variables, treatments, and outcomes
# We use the updated dataframe columns after the log transformation
exclude_from_proxies = exclude_cols + ['log_transport_co2']
numeric_df = df.select_dtypes(include=[np.number])
X_covariates = [col for col in numeric_df.columns if col not in exclude_from_proxies]

# 1. Deterministic Time Trend (t)
# Standardized so the first year in the panel equals 1
df['time_trend'] = df['year'] - df['year'].min() + 1

# 2. Cross-sectional averages (within-city over time) -> \bar{X}_i
city_averages = df.groupby('city_id')[X_covariates].transform('mean')
df = df.join(city_averages.add_suffix('_city_avg'))

# 3. Time series averages (within-time across cities) -> \bar{X}_t
time_averages = df.groupby('year')[X_covariates].transform('mean')
df = df.join(time_averages.add_suffix('_time_avg'))

# 4. Initial conditions (X_{i,0} and Y_{i,0})
# We must sort the dataframe chronologically to ensure 'first' captures the true baseline year
df = df.sort_values(by=['city_id', 'year']).reset_index(drop=True)

# Covariate initial conditions
initial_X = df.groupby('city_id')[X_covariates].transform('first')
df = df.join(initial_X.add_suffix('_initial'))

# Outcome initial conditions
df['log_transport_co2_initial'] = df.groupby('city_id')['log_transport_co2'].transform('first')

print("Success: Generated time trends, cross-sectional averages, time series averages, and initial conditions.")

# ---------------------------------------------------------
# 8. Export Final Cleaned Data
# ---------------------------------------------------------
save_dir_data = 'Data'
os.makedirs(save_dir_data, exist_ok=True)
file_path = os.path.join(save_dir_data, 'urban_emissions_panel_cleaned.csv')

df.to_csv(file_path, index=False)
print(f"Success: Final transformed dataset securely saved to {file_path}")