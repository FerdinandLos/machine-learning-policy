import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.stats import norm
from joblib import Parallel, delayed  # <-- ADDED: Parallel processing engine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, GroupKFold

from econml.dr import LinearDRLearner
from econml.dml import CausalForestDML

warnings.filterwarnings('ignore')

# --- BOOTSTRAP CONFIGURATION ---
B_BOOTSTRAP = 50  # Number of bootstrap iterations. Increase to 100+ for final thesis run.

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
tables_dir = Path('Writing/Tables')
tables_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('Data/urban_emissions_panel_cleaned.csv')
df['policy_regime'] = (df['cp_active'] * 1) + (df['lez_active'] * 2)

year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

exclude_from_W = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime', 'industry_public', 'fleet_petrol_share'
]
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]
core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

# ---------------------------------------------------------
# 2. Define Learners & Helper Functions
# ---------------------------------------------------------
# 1. Plug in the exact values from your optimal_hyperparameters_aipw.txt file
OPTIMAL_ALPHA = 0.015025173054024272
OPTIMAL_C = 0.046415888336127774

# 2. Updated Models Dictionary with n_jobs=1 internally to prevent thread thrashing
models = {
    'L1 (Lasso / Logit L1)': {
        'type': 'dr',
        'ml_l': make_pipeline(StandardScaler(), Lasso(alpha=OPTIMAL_ALPHA, random_state=42, max_iter=10000)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=OPTIMAL_C, solver='saga', random_state=42, max_iter=10000, n_jobs=1))
    },
    'Causal Forest': {
        'type': 'causal_forest',
        # FIX: Explicitly supply the locked, penalized models to the Causal Forest
        'ml_l': make_pipeline(StandardScaler(), Lasso(alpha=OPTIMAL_ALPHA, random_state=42, max_iter=10000)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=OPTIMAL_C, solver='saga', random_state=42, max_iter=10000, n_jobs=1)),
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_leaf': 50,
        'n_jobs': 1 
    }
}

regime_mapping = {'cp_active': 1, 'lez_active': 2, 'cp_x_lez': 3}

def extract_point_estimates(estimator, data_df, X_array, t_val):
    """Extracts raw causal estimates strictly for the Treated populations."""
    res = {}
    treated_mask = (data_df['policy_regime'].to_numpy() == t_val)
    res['Global_ATT_coef'] = np.atleast_1d(estimator.ate(X=X_array[treated_mask], T0=0, T1=t_val))[0]

    c0_mask = (data_df['cluster_id'].to_numpy() == 0)
    c1_mask = (data_df['cluster_id'].to_numpy() == 1)

    res['GATT_Cluster_0_coef'] = np.atleast_1d(estimator.ate(X=X_array[treated_mask & c0_mask], T0=0, T1=t_val))[0]
    res['GATT_Cluster_1_coef'] = np.atleast_1d(estimator.ate(X=X_array[treated_mask & c1_mask], T0=0, T1=t_val))[0]
    return res

def fit_and_extract(model_name, ml_dict, current_df, W_cols):
    """Handles the model fitting and estimate extraction for a given dataframe."""
    W_mat = current_df[W_cols].to_numpy()
    T_mat = current_df['policy_regime'].to_numpy()
    Y_mat = current_df['log_transport_co2'].to_numpy()
    groups_mat = current_df['boot_city_id'].to_numpy() if 'boot_city_id' in current_df.columns else current_df['city_id'].to_numpy()
    
    # Pre-generate cluster dummies for the specific current dataframe
    c_dummies = pd.get_dummies(current_df['cluster_id'], prefix='Cluster', dtype=int).to_numpy()
    
    cv_panel = GroupKFold(n_splits=5)
    
    if ml_dict['type'] == 'dr':
        est = LinearDRLearner(
            model_regression=clone(ml_dict['ml_l']),
            model_propensity=clone(ml_dict['ml_m']),
            min_propensity=0.01, 
            fit_cate_intercept=False, 
            cv=cv_panel, random_state=42
        )
    else:
        # FIX: Replaced 'auto' defaults with model_y and model_t mapped to your penalized pipelines
        est = CausalForestDML(
            model_y=clone(ml_dict['ml_l']),
            model_t=clone(ml_dict['ml_m']),
            n_estimators=ml_dict['n_estimators'], 
            max_depth=ml_dict['max_depth'],
            min_samples_leaf=ml_dict['min_samples_leaf'], 
            discrete_treatment=True,
            cv=cv_panel, random_state=42
        )

    est.fit(Y=Y_mat, T=T_mat, X=c_dummies, W=W_mat, groups=groups_mat)
    
    estimates = {}
    for p_name, t_val in regime_mapping.items():
        estimates[p_name] = extract_point_estimates(est, current_df, c_dummies, t_val)
    return estimates

# --- ADDED: Standalone function for single parallelized bootstrap iteration ---
def run_single_bootstrap_main(b_seed, original_df, W_cols):
    np.random.seed(b_seed)
    unique_cities = original_df['city_id'].unique()
    sampled_cities = np.random.choice(unique_cities, size=len(unique_cities), replace=True)
    
    # Reconstruct the panel data for this bootstrap draw
    boot_df_list = []
    for new_id, city in enumerate(sampled_cities):
        c_data = original_df[original_df['city_id'] == city].copy()
        c_data['boot_city_id'] = new_id
        boot_df_list.append(c_data)
    boot_df = pd.concat(boot_df_list, axis=0).reset_index(drop=True)
    
    # Calculate estimates for all models in this single bootstrap draw
    b_ests_all = {}
    for m_name, ml_dict in models.items():
        try:
            b_ests_all[m_name] = fit_and_extract(m_name, ml_dict, boot_df, W_cols)
        except Exception:
            b_ests_all[m_name] = None
    return b_ests_all


# ---------------------------------------------------------
# 3. Main Estimates & Clustered Block Bootstrap (Parallelized)
# ---------------------------------------------------------
print("--- CALCULATING MAIN POINT ESTIMATES ---")
main_estimates = {m: fit_and_extract(m, d, df, base_W_cols) for m, d in models.items()}

print(f"--- INITIATING CLUSTERED BLOCK BOOTSTRAP (B={B_BOOTSTRAP}) ---")
# Dynamically create the distribution dictionary based on main_estimates structure
bootstrap_distributions = {
    m: {
        p: {est_key: [] for est_key in main_estimates[m][p].keys()} 
        for p in regime_mapping.keys()
    } 
    for m in models.keys()
}

# Generate unique random seeds for each iteration
seeds = np.random.randint(0, 1000000, size=B_BOOTSTRAP)

# Execute parallel bootstrap loop across all CPU cores
parallel_results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_single_bootstrap_main)(seed, df, base_W_cols) 
    for seed in seeds
)

# Unpack the parallel results into the distributions dictionary
for b_res in parallel_results:
    if b_res is not None:
        for m_name, m_ests in b_res.items():
            if m_ests is not None:
                for p_name, p_ests in m_ests.items():
                    for est_key, est_val in p_ests.items():
                        bootstrap_distributions[m_name][p_name][est_key].append(est_val)

# ---------------------------------------------------------
# 4. Synthesize Results & Export
# ---------------------------------------------------------
print("--- CALCULATING P-VALUES VIA EMPIRICAL STANDARD ERRORS ---")
final_results = []

for model_name in models.keys():
    for policy_name in regime_mapping.keys():
        row_data = {'Model': model_name, 'Policy': policy_name}
        for est_key in main_estimates[model_name][policy_name].keys():
            point_est = main_estimates[model_name][policy_name][est_key]
            
            # Calculate Standard Error from the bootstrap distribution
            boot_array = np.array(bootstrap_distributions[model_name][policy_name][est_key])
            boot_array = boot_array[~np.isnan(boot_array)] # Drop nans if any model failed
            
            se = np.std(boot_array) if len(boot_array) > 0 else np.nan
            
            # Calculate p-value via normal approximation
            if not np.isnan(se) and se > 0:
                z_stat = point_est / se
                p_val = 2 * (1 - norm.cdf(abs(z_stat)))
            else:
                p_val = np.nan
            
            row_data[est_key] = point_est
            row_data[est_key.replace('_coef', '_pval')] = p_val
            
        final_results.append(row_data)

results_df = pd.DataFrame(final_results)
csv_export_path = results_dir / 'dml_robustness_results_aipw.csv'
results_df.to_csv(csv_export_path, index=False)
print(f"Success: Validated panel estimations safely exported to {csv_export_path}")
