import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.stats import norm
from joblib import Parallel, delayed  # <-- ADDED: Parallel processing engine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression # <-- Swapped from CV
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from econml.dr import LinearDRLearner

warnings.filterwarnings('ignore')

# --- BOOTSTRAP CONFIGURATION ---
B_BOOTSTRAP = 50 

# ---------------------------------------------------------
# 1. System & Data Setup
# ---------------------------------------------------------
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

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
regime_mapping = {'cp_active': 1, 'lez_active': 2, 'cp_x_lez': 3}
trim_thresholds = [0.05, 0.10]
model_name = 'L1 (Lasso / Logit L1)'

# ---------------------------------------------------------
# 2. Hyperparameter Locking & Pipeline Setup
# ---------------------------------------------------------
# Plug in the exact values from your optimal_hyperparameters_aipw.json file
import json

# Dynamically load the optimal hyperparameters from the tuning script
hyperparam_path = results_dir / 'optimal_hyperparameters_aipw.json'

try:
    with open(hyperparam_path, 'r') as f:
        optimal_params = json.load(f)
        
    OPTIMAL_ALPHA = optimal_params['OPTIMAL_ALPHA']
    OPTIMAL_C = optimal_params['OPTIMAL_C']
    
    print(f"[SETUP] Successfully loaded tuned hyperparameters:")
    print(f"        Alpha: {OPTIMAL_ALPHA}")
    print(f"        C: {OPTIMAL_C}")
    
except FileNotFoundError:
    raise FileNotFoundError(
        "Hyperparameter JSON not found. Please run the hyperparameter tuning script "
        "first to generate 'optimal_hyperparameters_aipw.json' in the Results directory."
    )   

# Updated with fixed hyperparameters, max_iter=10000 for convergence, and n_jobs=1 to avoid thread thrashing
ml_l = make_pipeline(StandardScaler(), Lasso(alpha=OPTIMAL_ALPHA, random_state=42, max_iter=10000))
ml_m = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=OPTIMAL_C, solver='saga', random_state=42, max_iter=10000, n_jobs=1))

def fit_and_extract_overlap(current_df, W_cols, trim):
    """Fits the model and extracts point estimates for a given dataframe and trim threshold."""
    W_mat = current_df[W_cols].to_numpy()
    T_mat = current_df['policy_regime'].to_numpy()
    Y_mat = current_df['log_transport_co2'].to_numpy()
    groups_mat = current_df['boot_city_id'].to_numpy() if 'boot_city_id' in current_df.columns else current_df['city_id'].to_numpy()
    c_dummies = pd.get_dummies(current_df['cluster_id'], prefix='Cluster', dtype=int).to_numpy()
    
    est = LinearDRLearner(
        model_regression=clone(ml_l), model_propensity=clone(ml_m),
        min_propensity=trim, fit_cate_intercept=False, 
        cv=GroupKFold(n_splits=5), random_state=42
    )
    est.fit(Y=Y_mat, T=T_mat, X=c_dummies, W=W_mat, groups=groups_mat)
    
    estimates = {}
    for p_name, t_val in regime_mapping.items():
        ate = np.atleast_1d(est.ate(X=c_dummies, T0=0, T1=t_val))[0]
        mask = (current_df['policy_regime'].to_numpy() == t_val)
        att = np.atleast_1d(est.ate(X=c_dummies[mask], T0=0, T1=t_val))[0]
        estimates[p_name] = {'Global_ATE_coef': ate, 'Global_ATT_coef': att}
    return estimates

# --- ADDED: Standalone function for single parallelized bootstrap iteration ---
def run_single_bootstrap(b_seed, original_df, W_cols, trim):
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
    
    try:
        b_ests = fit_and_extract_overlap(boot_df, W_cols, trim)
        return b_ests
    except Exception:
        return None

# ---------------------------------------------------------
# 3. Sensitivity Loop & Parallel Bootstrap
# ---------------------------------------------------------
print("--- INITIATING OVERLAP SENSITIVITY (WITH PARALLEL BOOTSTRAP) ---")
final_results = []

for trim in trim_thresholds:
    print(f"\n>> Threshold: {trim}")
    try:
        # 1. Main Point Estimates
        main_ests = fit_and_extract_overlap(df, base_W_cols, trim)
        
        # 2. Parallel Bootstrap Loop for Empirical Standard Errors
        boot_dists = {p: {'Global_ATE_coef': [], 'Global_ATT_coef': []} for p in regime_mapping.keys()}
        
        # Generate unique random seeds for each iteration
        seeds = np.random.randint(0, 1000000, size=B_BOOTSTRAP)
        
        # joblib.Parallel distributes the iterations across all CPU cores simultaneously
        parallel_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(run_single_bootstrap)(seed, df, base_W_cols, trim) 
            for seed in seeds
        )
        
        # Unpack the parallel results into the distributions dictionary
        for b_ests in parallel_results:
            if b_ests is not None:
                for p in regime_mapping.keys():
                    boot_dists[p]['Global_ATE_coef'].append(b_ests[p]['Global_ATE_coef'])
                    boot_dists[p]['Global_ATT_coef'].append(b_ests[p]['Global_ATT_coef'])
                
        # 3. Synthesize and Calculate p-values
        for p_name in regime_mapping.keys():
            row_data = {'Model': model_name, 'Trim_Threshold': trim, 'Policy': p_name}
            for est_key in ['Global_ATE_coef', 'Global_ATT_coef']:
                pt_est = main_ests[p_name][est_key]
                
                boot_array = np.array(boot_dists[p_name][est_key])
                boot_array = boot_array[~np.isnan(boot_array)]
                
                se = np.std(boot_array) if len(boot_array) > 0 else np.nan
                pval = 2 * (1 - norm.cdf(abs(pt_est / se))) if se > 0 else np.nan
                
                row_data[est_key] = pt_est
                row_data[est_key.replace('_coef', '_pval')] = pval
                
            final_results.append(row_data)
            
    except Exception as e:
        print(f"   [!] Failed for threshold {trim}: {e}")

# ---------------------------------------------------------
# 4. Export the results
# ---------------------------------------------------------
sensitivity_df = pd.DataFrame(final_results)
export_path = results_dir / 'sensitivity_overlap_trimming.csv'
sensitivity_df.to_csv(export_path, index=False)
print(f"\nSuccess: Overlap data safely exported to {export_path}")