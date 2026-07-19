import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from doubleml import DoubleMLData, DoubleMLPLR


# Load the urban emissions panel dataset
csv_path = Path(__file__).resolve().parents[1] / "Data" / "urban_emissions_panel.csv"
df = pd.read_csv(csv_path)

print(df.head())

# --------------------------
# I. Find the optimal K clusters
# --------------------------

# 1. Isolate exogenous variables and account for panel structure
# Define columns that must NOT be used for clustering
exclude_cols = ['year', 'transport_co2', 'total_co2', 'cp_active', 'lez_active', 
                'cp_impl_year', 'lez_impl_year', 'cp_announce_year', 'lez_announce_year', 'country_id']

# Drop excluded columns and aggregate to one row per city (mean over time)
city_features = df.drop(columns=exclude_cols).groupby('city_id').mean()

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
summary_table.to_latex('Writing/Tables/cluster_summary.tex')
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

# -------------
# MAIN ANALYSIS
# -------------

# ---------------------------------------------------------
# 1. Data Preparation & Feature Engineering
# ---------------------------------------------------------

# Create the synergy interaction term for Model 1 (Question 3)
df['cp_x_lez'] = df['cp_active'] * df['lez_active']

# Create the heterogeneity interaction terms for Model 2 (Question 2)
# Assuming 'cluster_id' is 0 for Sprawling Hubs and 1 for Dense Metropolises
df['cp_x_type1'] = df['cp_active'] * df['cluster_id']
df['lez_x_type1'] = df['lez_active'] * df['cluster_id']

# Create year dummies to account for macro time trends in the panel
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define the confounder matrix (X)
# We exclude the outcome, the treatments, the IDs, and the raw year/announcement columns
exclude_from_X = ['city_id', 'year', 'transport_co2', 'total_co2', 
                  'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
                  'cp_announce_year', 'lez_announce_year', 'country_id',
                  'cp_x_lez', 'cp_x_type1', 'lez_x_type1', 'cluster_id']

X_cols = [col for col in df.columns if col not in exclude_from_X]

# ---------------------------------------------------------
# 2. Setup Machine Learning Learners
# ---------------------------------------------------------

# We use Random Forests for the nuisance functions E[Y|X] and E[D|X]
# Setting a fixed random_state ensures reproducibility for your report
ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42) # classifier because it's a binary treatment dummy

# ---------------------------------------------------------
# 3. Model 1: Main Effects and Synergy
# ---------------------------------------------------------
print("Estimating Model 1: Main Effects and Synergy...")

# Define the treatment variables (D)
D_cols_m1 = ['cp_active', 'lez_active', 'cp_x_lez']

# Initialize the DoubleMLData object
dml_data_m1 = DoubleMLData(df, y_col='transport_co2', d_cols=D_cols_m1, x_cols=X_cols)

# Initialize and fit the Partially Linear Regression (PLR) model
# We use Neyman-orthogonal scores ('partialling out') natively 
dml_plr_m1 = DoubleMLPLR(dml_data_m1, 
                         ml_l=clone(ml_l), 
                         ml_m=clone(ml_m), 
                         n_folds=5)
dml_plr_m1.fit()

print(dml_plr_m1.summary)

# ---------------------------------------------------------
# 4. Model 2: Heterogeneity by City Type
# ---------------------------------------------------------
print("\nEstimating Model 2: Heterogeneity by City Type...")

# Define the treatment variables (D) including the cluster interactions
D_cols_m2 = ['cp_active', 'lez_active', 'cp_x_type1', 'lez_x_type1']

# Initialize the DoubleMLData object
dml_data_m2 = DoubleMLData(df, y_col='transport_co2', d_cols=D_cols_m2, x_cols=X_cols)

# Initialize and fit the model
dml_plr_m2 = DoubleMLPLR(dml_data_m2, 
                         ml_l=clone(ml_l), 
                         ml_m=clone(ml_m), 
                         n_folds=5)
dml_plr_m2.fit()

print(dml_plr_m2.summary)

# ---------------------------------------------------------
# 1. Setup the Linear Nuisance Learners with Pipelines
# ---------------------------------------------------------

# Pipeline for ml_l: Standardize features -> Lasso Cross-Validation
# LassoCV automatically searches for the optimal penalty parameter (alpha)
lasso_learner = make_pipeline(
    StandardScaler(),
    LassoCV(cv=5, random_state=42, max_iter=10000)
)

# Pipeline for ml_m: Standardize features -> Penalized Logistic Regression
# We use penalty='l1' and solver='liblinear' to apply the exact Lasso equivalent for classification
logistic_learner = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', 
                         random_state=42, max_iter=10000)
)

# ---------------------------------------------------------
# 2. Run the Sensitivity Analysis on Model 1
# ---------------------------------------------------------
print("Running Sensitivity Analysis: Model 1 with Penalized Linear Models...")

# Use the exact same D_cols_m1 and X_cols from your previous script
# D_cols_m1 = ['cp_active', 'lez_active', 'cp_x_lez']

dml_data_m1_sens = DoubleMLData(df, y_col='transport_co2', d_cols=D_cols_m1, x_cols=X_cols)

# Initialize and fit the Partially Linear Regression (PLR) with the linear learners
dml_plr_m1_sens = DoubleMLPLR(dml_data_m1_sens, 
                              ml_l=lasso_learner, 
                              ml_m=logistic_learner, 
                              n_folds=5)
dml_plr_m1_sens.fit()

# ---------------------------------------------------------
# 3. Output Comparison
# ---------------------------------------------------------
print("\n--- BASELINE RESULTS (RANDOM FOREST) ---")
print(dml_plr_m1.summary) # Assuming dml_plr_m1 from previous run is still in memory

print("\n--- SENSITIVITY RESULTS (LASSO / LOGISTIC) ---")
print(dml_plr_m1_sens.summary)