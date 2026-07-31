import pandas as pd
import numpy as np
import warnings
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import GroupKFold

# Suppress convergence warnings during the grid search
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

print("Loading dataset for hyperparameter tuning...")
df = pd.read_csv('Data/urban_emissions_panel_cleaned.csv')

df['policy_regime'] = (df['cp_active'] * 1) + (df['lez_active'] * 2)

year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

exclude_from_W = [
    # Outcomes, Treatments, IDs
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime', 
    
    # Post-Treatment Mediators & Feedback Loops (Rely on Mundlak means instead)
    'pm25', 'fleet_diesel_share', 'fleet_electric_share', 'fleet_petrol_share',
    'public_transit_score', 'road_km_pc', 'industry_public',
    'logistics_activity', 'industry_logistics', 'tourism_intensity', # <-- NEW
    'political_green', 'ngo_environment_index', 'electoral_competitiveness' # <-- NEW
]
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]

# ---------------------------------------------------------
# 2. Strict Panel Cross-Validation Setup
# ---------------------------------------------------------
W_matrix = df[base_W_cols].to_numpy()
Y_arr = df['log_transport_co2'].to_numpy()
T_arr = df['policy_regime'].to_numpy()
city_groups = df['city_id'].to_numpy()

# Scale data outside the pipeline to ensure clean CV routing
scaler = StandardScaler()
W_scaled = scaler.fit_transform(W_matrix)

# Create strict grouped splits so cities are never mixed across train/validation sets
group_kfold = GroupKFold(n_splits=5)
cv_splits = list(group_kfold.split(W_scaled, Y_arr, city_groups))

print(f"Configuring strict GroupKFold pipelines with {len(base_W_cols)} baseline covariates...")

# Pass the pre-computed splits to cv=cv_splits
lasso_cv = LassoCV(cv=cv_splits, random_state=42, max_iter=10000, n_jobs=-1)

# Force multinomial evaluation to get a single global penalty
logit_cv = LogisticRegressionCV(
    cv=cv_splits, penalty='l1', solver='saga', scoring='neg_log_loss', 
    random_state=42, max_iter=10000, n_jobs=-1
)

# ---------------------------------------------------------
# 3. Execute Grouped Hyperparameter Search
# ---------------------------------------------------------
print("\n>> Tuning LassoCV for Outcome Model (Log Transport CO2)...")
lasso_cv.fit(W_scaled, Y_arr)
best_alpha_y = lasso_cv.alpha_
print(f"   Optimal alpha found: {best_alpha_y}")

print("\n>> Tuning LogisticRegressionCV for Treatment Model (Policy Regime)...")
logit_cv.fit(W_scaled, T_arr)
best_C_t = logit_cv.C_[0]
print(f"   Optimal C found: {best_C_t}")

# ---------------------------------------------------------
# 4. Export Results
# ---------------------------------------------------------
hyperparam_txt_path = results_dir / 'optimal_hyperparameters_aipw.txt'
hyperparam_json_path = results_dir / 'optimal_hyperparameters_aipw.json'

with open(hyperparam_txt_path, 'w') as f:
    f.write("--- LASSO (L1) OPTIMAL HYPERPARAMETERS (STRICT GROUP K-FOLD) ---\n\n")
    f.write(f"Outcome Model (Lasso) Selected alpha (regularization penalty): {best_alpha_y}\n")
    f.write(f"Treatment Model (LogisticRegression L1) Selected C (inverse penalty): {best_C_t}\n")

hyperparam_dict = {
    'OPTIMAL_ALPHA': float(best_alpha_y),
    'OPTIMAL_C': float(best_C_t)
}

with open(hyperparam_json_path, 'w') as f:
    json.dump(hyperparam_dict, f, indent=4)

print(f"\nSuccess: Hyperparameters saved to {hyperparam_txt_path} and {hyperparam_json_path}")