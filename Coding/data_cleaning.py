import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 0. Global Academic Visual Styling
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

csv_path = Path(__file__).resolve().parents[1] / "Data" / "urban_emissions_panel.csv"
df = pd.read_csv(csv_path)

print(df.head())

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(df['transport_co2'].dropna(), bins=40, color='#a6cee3', edgecolor='black', alpha=0.8)
ax1.set_title('Raw Distribution of Transport $CO_2$', pad=15, fontweight='bold')
ax1.set_xlabel('Transport $CO_2$ (kt)')
ax1.set_ylabel('Frequency')

ax2.hist(np.log(df['transport_co2'].dropna()), bins=40, color='#1f78b4', edgecolor='black', alpha=0.8)
ax2.set_title('Log-Transformed Distribution', pad=15, fontweight='bold')
ax2.set_xlabel('Log(Transport $CO_2$)')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'co2_distribution.pdf'), format='pdf', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 3b. Convert Categorical Variables to Dummies
# ---------------------------------------------------------
# Convert latitude_zone (1, 2, 3) into binary indicators to prevent continuous treatment
df = pd.get_dummies(df, columns=['latitude_zone'], prefix='lat_zone', drop_first=True, dtype=int)

#---------------------------------------------------------
# 4. Check all variables for skewed distributions
#---------------------------------------------------------
exclude_cols = [
    'city_id', 'year', 'country_id', 'cluster_id', 'lat_zone_2', 'lat_zone_3',
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

if not highly_skewed_cols.empty:
    cols_to_plot = highly_skewed_cols.index.tolist()
    n_cols = 3
    n_rows = math.ceil(len(cols_to_plot) / n_cols)
    
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    
    if n_rows > 1:
        axes = axes.flatten()
    elif len(cols_to_plot) > 1:
        axes = axes
    else:
        axes = [axes] 
        
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(df[col].dropna(), bins=30, color='#7570b3', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col}\n(Skew: {highly_skewed_cols[col]:.2f})', pad=10, fontweight='bold')
        axes[i].set_yticks([])
        
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'skewness_diagnostics.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.rcParams.update({'font.size': 12})

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

columns_to_drop = ['transport_co2'] + covariates_to_log
df = df.drop(columns=columns_to_drop)

# ---------------------------------------------------------
# 6. Categorize and Construct Time-Invariant Proxies
# ---------------------------------------------------------
print("\n--- CONSTRUCTING PANEL HETEROGENEITY PROXIES ---")

# 1. Explicitly define your strict time-invariant variables
time_invariant_cols = ['log_area_km2', 'elevation', 'coastal', 'lat_zone_2', 'lat_zone_3']
time_invariant_cols = [c for c in time_invariant_cols if c in df.columns]

# 2. Define base exclusions (Identifiers, Policy Timings, and Outcomes)
base_exclusions = [
    'city_id', 'year', 'country_id', 'cluster_id',
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year',
    'cp_announce_year', 'lez_announce_year', 
    'transport_co2', 'log_transport_co2', 'total_co2', 'log_total_co2'
]

# 3. Dynamically capture ALL remaining numeric columns as time-varying covariates
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
time_varying_cols = [col for col in numeric_cols 
                     if col not in base_exclusions 
                     and col not in time_invariant_cols]

# Ensure chronological order
df = df.sort_values(by=['city_id', 'year']).reset_index(drop=True)

# 4. Generate _initial variables strictly for time-invariant features
for col in time_invariant_cols:
    df[f'{col}_initial'] = df.groupby('city_id')[col].transform('first')

print(f"Success: Isolated {len(time_invariant_cols)} time-invariant features.")

# ---------------------------------------------------------
# 7. PROPERLY-TIMED MUNDLAK DEVICE IMPLEMENTATION
# ---------------------------------------------------------
print(f"Constructing Mundlak device for {len(time_varying_cols)} time-varying covariates...")

df['first_policy_year'] = df[['cp_impl_year', 'lez_impl_year']].min(axis=1)

pre_treatment_mask = (df['year'] < df['first_policy_year']) | df['first_policy_year'].isna()
pre_df = df[pre_treatment_mask]

# Calculate the pre-treatment means for the massive list of time-varying variables
mundlak_means = pre_df.groupby('city_id')[time_varying_cols].mean()
mundlak_means = mundlak_means.add_suffix('_pre_mean')
df = df.merge(mundlak_means, on='city_id', how='left')

# Safety fallback for immediate adopters
for col in mundlak_means.columns:
    if df[col].isna().any():
        base_col = col.replace('_pre_mean', '')
        first_obs = df.sort_values(['city_id', 'year']).groupby('city_id')[base_col].transform('first')
        df[col] = df[col].fillna(first_obs)

df = df.drop(columns=['first_policy_year'])

print("Success: Full Mundlak proxy matrix generated.")

# ---------------------------------------------------------
# Check and Remove Zero-Variance Variables (e.g., Climate Pact)
# ---------------------------------------------------------
# Define the specific column to investigate
pact_col = 'national_climate_pact_pre_mean'

if pact_col in df.columns:
    # Check if the maximum value is 0 (meaning all rows are 0)
    if df[pact_col].max() == 0:
        print(f"\n[DIAGNOSTIC] Confirmed: '{pact_col}' is exactly 0 for all cities.")
        print("This indicates the pact was non-existent during the pre-treatment period.")
        print(f"-> Dropping '{pact_col}' from the dataset.")
        
        # Drop the pre_mean column
        df = df.drop(columns=[pact_col])
        
        # Optionally, drop the raw base column as well if you don't need it for later estimations
        if 'national_climate_pact' in df.columns:
            df = df.drop(columns=['national_climate_pact'])
            
    else:
        print(f"\n[DIAGNOSTIC] '{pact_col}' contains non-zero values.")
        print("Mean value across cities:", df[pact_col].mean())

# ---------------------------------------------------------
# 8. Find the Optimal K Clusters (Strictly Pre-Treatment via Proxies)
# ---------------------------------------------------------
print("\n--- INITIATING K-MEANS CLUSTERING ---")

# 1. Extract exactly one row per city, as the proxies are time-invariant
city_features = df.drop_duplicates('city_id').set_index('city_id')

# 2. Filter strictly for the newly created pre-treatment columns
# We exclude the CO2 outcomes to prevent clustering on the dependent variables
cluster_cols = [col for col in city_features.columns 
                if (col.endswith('_pre_mean') or col.endswith('_initial')) 
                and 'co2' not in col 
                and 'lat_zone' not in col]

city_features_cluster = city_features[cluster_cols].copy()

# 3. Standardize and evaluate clusters
X_scaled = StandardScaler().fit_transform(city_features_cluster)
k_range = range(2, 10)
inertias, silhouettes = [], []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(k_range, inertias, marker='s', color='#d95f02', linestyle='-', linewidth=2, markersize=7)
ax1.set_title('Elbow Method (Inertia)', pad=15, fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')

ax2.plot(k_range, silhouettes, marker='o', color='#1b9e77', linestyle='-', linewidth=2, markersize=7)
ax2.set_title('Silhouette Analysis', pad=15, fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')

plt.tight_layout()
pdf_path = os.path.join(save_dir, 'kmeans_evaluation.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()

# 4. Apply the optimal K = 2
optimal_k = 2
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(X_scaled)
city_features_cluster['cluster_id'] = final_kmeans.labels_

# 5. Map the cluster IDs back to the main panel dataset
cluster_mapping = dict(zip(city_features_cluster.index, final_kmeans.labels_))
df['cluster_id'] = df['city_id'].map(cluster_mapping)

# ----------
# Show how the two types differ
# ----------

X_scaled_df = pd.DataFrame(X_scaled, columns=city_features_cluster.drop(columns='cluster_id').columns)
X_scaled_df['cluster_id'] = final_kmeans.labels_
scaled_means = X_scaled_df.groupby('cluster_id').mean()

# Calculate absolute differences for all features
std_diff_absolute = abs(scaled_means.loc[1] - scaled_means.loc[0])

# --- NEW: Extract Top 8 (Largest differences) and Bottom 4 (Smallest differences / Zeros) ---
top_10_features = std_diff_absolute.sort_values(ascending=False).head(10).index.tolist()
bottom_4_features = std_diff_absolute.sort_values(ascending=True).head(4).index.tolist()

# Combine them into a single list of 12 features
combined_features = top_10_features + bottom_4_features

# Build the Visualization using the combined features
scaled_diff = (scaled_means.loc[1] - scaled_means.loc[0])[combined_features]

# 1. Initialize an empty dictionary
plot_names = {}

# 2. Assign the clean names
plot_names['log_pop_density_pre_mean'] = "Log Population per $km^2$"
plot_names['public_transit_score_pre_mean'] = "Public Transit Score"
plot_names['log_population_pre_mean'] = "Log Population"
plot_names['library_count_pre_mean'] = "No. of Libraries"
plot_names['fleet_petrol_share_pre_mean'] = "Petrol Vehicle Share"
plot_names['fleet_diesel_share_pre_mean'] = "Diesel Vehicle Share"
plot_names['log_area_km2_initial'] = "Log Area ($km^2$)"
plot_names['fleet_electric_share_pre_mean'] = "Electric Vehicle Share"
plot_names['industry_logistics_pre_mean'] = "Logistics Employment Share"
plot_names['political_green_pre_mean'] = "Green Party Vote Share"
plot_names['ngo_environment_index_pre_mean'] = "Environmental NGO Index"
plot_names['sister_city_count_pre_mean'] = "No. of Sister-Cities"
plot_names['log_electricity_price_pre_mean'] = "Log Electricity Price"
plot_names['renewable_electricity_share_pre_mean'] = "Renewable Electricity Share"
plot_names['streetlight_density_pre_mean'] = "Streetlights per km of Road"
plot_names['flagpole_count_pre_mean'] = "No. of Flagpoles on Public Buildings"
plot_names['unemployment_pre_mean'] = "Unemployment rate"
plot_names['industry_public_pre_mean'] = "Public Sector Employment Share"
plot_names['log_fuel_price_pre_mean'] = "Log Average Fuel Price"
plot_names['tourism_intensity_pre_mean'] = "Tourism Intensity"
plot_names['education_share_pre_mean'] = "Tertiary Education Share"

# 3. CRITICAL: Actually apply the renaming map to the Series
scaled_diff = scaled_diff.rename(index=plot_names)

# This mathematically guarantees the bars taper down nicely, with the 0s at the very bottom
scaled_diff = scaled_diff.reindex(scaled_diff.abs().sort_values(ascending=False).index)

# Create figure
fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#1f78b4' if val > 0 else '#a6cee3' for val in scaled_diff]
scaled_diff.plot(kind='barh', color=colors, edgecolor='black', ax=ax)

# --- NEW: Add a subtle horizontal separator line ---
# Because pandas plots the 12 bars at y-coordinates 0 through 11, 
# and we are about to invert the y-axis, placing a line at y=7.5 
# perfectly divides the 8th bar and the 9th bar.
ax.axhline(y=9.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Add text annotations directly onto the chart to explain the sections
# The x-coordinate is set near the middle/right of the chart area. Adjust x as needed based on your actual data range.
ax.text(x=1, y=9.35, s='Structural Drivers', color='black', fontsize=10, fontstyle='italic', ha='center')
ax.text(x=1, y=9.85, s='Orthogonal Baselines', color='black', fontsize=10, fontstyle='italic', ha='center')

# Updated Title to reflect the academic narrative
ax.set_title('Top Structural Drivers vs. Orthogonal Controls', pad=15, fontweight='bold')
ax.set_xlabel('Difference in Standard Deviations\n(Dense Metropolises - Sprawling Hubs)')
plt.gca().invert_yaxis() 
plt.tight_layout()

plt.savefig(os.path.join(save_dir, 'cluster_differences.pdf'), format='pdf', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Print Scaled Differences Across ALL Features in Console
# ---------------------------------------------------------
# 1. Calculate the directional scaled difference for ALL features (Dense - Sprawling)
full_scaled_diff = scaled_means.loc[1] - scaled_means.loc[0]

# 2. Sort by absolute magnitude so the most distinguishing features appear at the top
full_scaled_diff_sorted = full_scaled_diff.reindex(full_scaled_diff.abs().sort_values(ascending=False).index)

# 3. Create a clean DataFrame for console display
all_features_df = pd.DataFrame({
    'Feature': full_scaled_diff_sorted.index,
    'Std_Dev_Diff (Dense - Sprawling)': full_scaled_diff_sorted.values
})

# Map clean descriptions if available
all_features_df['Description'] = all_features_df['Feature'].fillna(all_features_df['Feature'])

print("\n" + "="*80)
print("FULL LIST OF SCALED DIFFERENCES ACROSS ALL CLUSTER FEATURES (in Std. Devs)")
print("="*80)
# pd.option_context ensures Pandas prints all rows without truncating the middle with '...'
with pd.option_context('display.max_rows', None, 'display.float_format', lambda x: f'{x:+.3f}'):
    print(all_features_df[['Description', 'Feature', 'Std_Dev_Diff (Dense - Sprawling)']].to_string(index=False))
print("="*80 + "\n")

# ---------------------------------------------------------
# Export LaTeX Table: Combined Treatment Assignment by Cluster
# ---------------------------------------------------------
print("Generating Combined Treatment by Cluster Cross-Tabulation...")

# 1. Aggregate to the unique city level
# 'max' ensures that if a city EVER had the policy active, it counts as 1
city_treatment = df.groupby('city_id').agg({
    'cluster_id': 'first',
    'cp_active': 'max',
    'lez_active': 'max'
}).reset_index()

# 2. Map the cluster IDs to your official thesis names
cluster_map = {
    0: 'Regional Hubs (0)',
    1: 'Dense Metropolises (1)'
}
city_treatment['Cluster'] = city_treatment['cluster_id'].map(cluster_map)

# 3. Create a mutually exclusive combined treatment category
def determine_treatment(row):
    if row['cp_active'] == 0 and row['lez_active'] == 0:
        return 'Never Treated'
    elif row['cp_active'] == 1 and row['lez_active'] == 0:
        return 'CP Only'
    elif row['cp_active'] == 0 and row['lez_active'] == 1:
        return 'LEZ Only'
    elif row['cp_active'] == 1 and row['lez_active'] == 1:
        return 'Synergy'
    return 'Unknown'

city_treatment['Treatment_State'] = city_treatment.apply(determine_treatment, axis=1)

# Force a specific logical column order for the final table
col_order = ['Never Treated', 'CP Only', 'LEZ Only', 'Synergy']
city_treatment['Treatment_State'] = pd.Categorical(city_treatment['Treatment_State'], categories=col_order, ordered=True)

# 4. Create the cross-tabulation with row and column totals
crosstab_combined = pd.crosstab(
    city_treatment['Cluster'], 
    city_treatment['Treatment_State'], 
    margins=True, 
    margins_name='Total'
)

# 5. Build the manual LaTeX table
latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Distribution of Climate Policy Treatments by Urban Typology}")
latex_lines.append(r"\label{tab:treatment_by_cluster}")
# lccccc creates 6 columns (1 string for cluster name, 5 numeric)
latex_lines.append(r"\begin{tabular}{lccccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Urban Typology & Never Treated & CP Only & LEZ Only & Synergy (CP $\times$ LEZ) & Total \\")
latex_lines.append(r"\midrule")

# Loop through the rows (excluding the last 'Total' row first)
for index, row in crosstab_combined.iloc[:-1].iterrows():
    latex_lines.append(f"{index} & {row['Never Treated']} & {row['CP Only']} & {row['LEZ Only']} & {row['Synergy']} & {row['Total']} \\\\")

# Add a midrule before the Total line to make it pop
latex_lines.append(r"\midrule")
total_row = crosstab_combined.iloc[-1]
latex_lines.append(f"\\textbf{{{total_row.name}}} & \\textbf{{{total_row['Never Treated']}}} & \\textbf{{{total_row['CP Only']}}} & \\textbf{{{total_row['LEZ Only']}}} & \\textbf{{{total_row['Synergy']}}} & \\textbf{{{total_row['Total']}}} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"") 
latex_lines.append(r"\vspace{1ex}")
latex_lines.append(r"{\raggedright \footnotesize \textit{Notes:} Table displays the count of unique cities.") 
latex_lines.append(r"A city is classified as ``Ever Treated'' if Congestion Pricing was active in any year during the sample period.\par}")
latex_lines.append(r"\end{table}")

# 6. Save file to disk
tables_dir = Path('Writing/Tables')
with open(tables_dir / 'treatment_by_cluster.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: treatment_by_cluster.tex saved.")

# ---------------------------------------------------------
# 9. Export Final Cleaned Data
# ---------------------------------------------------------
variable_names = df.columns.tolist()

print(variable_names)

save_dir_data = 'Data'
os.makedirs(save_dir_data, exist_ok=True)
file_path = os.path.join(save_dir_data, 'urban_emissions_panel_cleaned.csv')

df.to_csv(file_path, index=False)
print(f"Success: Final transformed dataset securely saved to {file_path}")