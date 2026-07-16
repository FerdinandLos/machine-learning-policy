import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# 4. Plot both metrics side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(k_range, inertias, marker='o')
ax1.set(title='Elbow Method (Inertia)', xlabel='Number of Clusters (K)', ylabel='Inertia')

ax2.plot(k_range, silhouettes, marker='o', color='orange')
ax2.set(title='Silhouette Analysis', xlabel='Number of Clusters (K)', ylabel='Silhouette Score')

plt.tight_layout()
plt.show()

# 5. Apply the optimal K (assuming visual inspection reveals K=3 is optimal)
optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(X_scaled)

# Assign the fixed cluster IDs back to the aggregated dataframe
city_features['cluster_id'] = final_kmeans.labels_

# Merge the fixed cluster IDs back to the original panel dataset based on city_id
df = df.merge(city_features[['cluster_id']], on='city_id', how='left')