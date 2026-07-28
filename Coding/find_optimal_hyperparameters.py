import pandas as pd
import numpy as np
import warnings
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline

# Suppress convergence warnings during the grid search
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. System Setup
# ---------------------------------------------------------
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Dynamic Control Setup
# ---------------------------------------------------------
print("Loading dataset for hyperparameter tuning...")
df = pd.read_csv('Data/urban_emissions_panel_cleaned.csv')

# Create the mutually exclusive categorical policy regime
df['policy_regime'] = (df['cp_active'] * 1) + (df['lez_active'] * 2)

# Generate year dummies to ensure time fixed effects are included in the penalty search
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define Base W (Covariates/Confounders without ANY policies)
exclude_from_W = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime', 'industry_public', 'fleet_petrol_share'
]
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]

# ---------------------------------------------------------
# 3. Configure the CV Pipelines
# ---------------------------------------------------------
print(f"Configuring CV pipelines with {len(base_W_cols)} baseline covariates...")

lasso_cv = make_pipeline(
    StandardScaler(), 
    LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
)

logit_cv = make_pipeline(
    StandardScaler(), 
    LogisticRegressionCV(
        cv=5, penalty='l1', solver='saga', scoring='neg_log_loss', 
        random_state=42, max_iter=10000, n_jobs=-1
    )
)

# ---------------------------------------------------------
# 4. Execute Full-Sample Hyperparameter Search
# ---------------------------------------------------------
W_matrix = df[base_W_cols].to_numpy()
Y_arr = df['log_transport_co2'].to_numpy()
T_arr = df['policy_regime'].to_numpy()

print("\n>> Tuning LassoCV for Outcome Model (Log Transport CO2)...")
lasso_cv.fit(W_matrix, Y_arr)
best_alpha_y = lasso_cv.steps[1][1].alpha_
print(f"   Optimal alpha found: {best_alpha_y}")

print("\n>> Tuning LogisticRegressionCV for Treatment Model (Policy Regime)...")
logit_cv.fit(W_matrix, T_arr)
# Multiclass LogisticRegressionCV returns an array of C_ values. We take the first one.
best_C_t = logit_cv.steps[1][1].C_[0]
print(f"   Optimal C found: {best_C_t}")

# ---------------------------------------------------------
# 5. Export Results
# ---------------------------------------------------------
hyperparam_txt_path = results_dir / 'optimal_hyperparameters_aipw.txt'
hyperparam_json_path = results_dir / 'optimal_hyperparameters_aipw.json'

# Export as Text (For human readability/thesis write-up)
with open(hyperparam_txt_path, 'w') as f:
    f.write("--- LASSO (L1) OPTIMAL HYPERPARAMETERS (FULL SAMPLE) ---\n\n")
    f.write(f"Outcome Model (Lasso) Selected alpha (regularization penalty): {best_alpha_y}\n")
    f.write(f"Treatment Model (LogisticRegression L1) Selected C (inverse penalty): {best_C_t}\n")

# Export as JSON (For programmatic loading in bootstrap scripts)
hyperparam_dict = {
    'OPTIMAL_ALPHA': float(best_alpha_y),
    'OPTIMAL_C': float(best_C_t)
}

with open(hyperparam_json_path, 'w') as f:
    json.dump(hyperparam_dict, f, indent=4)

print(f"\nSuccess: Hyperparameters saved to {hyperparam_txt_path} and {hyperparam_json_path}")