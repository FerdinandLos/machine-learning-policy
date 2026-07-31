import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.stats import norm
from joblib import Parallel, delayed  # <-- ADDED: Parallel processing engine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

# EconML Imports for AIPW Doubly Robust Estimation
from econml.dr import LinearDRLearner
from econml.dml import CausalForestDML

# Suppress warnings for clean console matrix outputs
warnings.filterwarnings('ignore')

# --- BOOTSTRAP CONFIGURATION ---
B_BOOTSTRAP = 50  # Number of iterations. Increase for final thesis run.

# ---------------------------------------------------------
# 1. System Setup
# ---------------------------------------------------------
tables_dir = Path('Writing/Tables')
tables_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Dynamic Control Setup
# ---------------------------------------------------------
df = pd.read_csv('Data/urban_emissions_panel_cleaned.csv')

# Create the mutually exclusive categorical policy regime
df['policy_regime'] = (df['cp_active'] * 1) + (df['lez_active'] * 2)

year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define Base W (Covariates/Confounders without ANY policies)
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

# --- BUG 3 FIX: Corrected list string ---
core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

# ---------------------------------------------------------
# 3. Define the Learners Grid & Helper Functions
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

# 2. Models Dictionary
models = {
    'L1 (Lasso / Logit L1)': {
        'type': 'dr',
        'ml_l': make_pipeline(StandardScaler(), Lasso(alpha=OPTIMAL_ALPHA, random_state=42, max_iter=10000)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=OPTIMAL_C, solver='saga', random_state=42, max_iter=10000, n_jobs=1))
    },
    'Causal Forest': {
        'type': 'causal_forest',
        'ml_l': make_pipeline(StandardScaler(), Lasso(alpha=OPTIMAL_ALPHA, random_state=42, max_iter=10000)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=OPTIMAL_C, solver='saga', random_state=42, max_iter=10000, n_jobs=1)),
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_leaf': 50,
        'n_jobs': 1 
    }
}

regime_mapping = {
    'cp_active': 1,
    'lez_active': 2,
    'cp_x_lez': 3
}

def fit_and_extract_placebo(ml_dict, current_df, W_cols, placebo_var):
    """Fits the model and extracts point estimates for a given dataframe."""
    W_mat = current_df[W_cols].to_numpy()
    T_mat = current_df['policy_regime'].to_numpy()
    Y_mat = current_df[placebo_var].to_numpy()
    
    # Use pseudo-IDs if bootstrapping, otherwise use real city_ids
    groups_mat = current_df['boot_city_id'].to_numpy() if 'boot_city_id' in current_df.columns else current_df['city_id'].to_numpy()
    
    # Dynamically generate cluster dummies for the current dataframe length
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
        ate = np.atleast_1d(est.ate(X=c_dummies, T0=0, T1=t_val))[0]
        mask = (current_df['policy_regime'].to_numpy() == t_val)
        att = np.atleast_1d(est.ate(X=c_dummies[mask], T0=0, T1=t_val))[0]
        estimates[p_name] = {'Global_ATE_coef': ate, 'Global_ATT_coef': att}
    return estimates

# --- ADDED: Standalone function for single parallelized bootstrap iteration ---
def run_single_bootstrap(b_seed, ml_dict, original_df, W_cols, placebo_var):
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
        b_ests = fit_and_extract_placebo(ml_dict, boot_df, W_cols, placebo_var)
        return b_ests
    except Exception:
        return None

# ---------------------------------------------------------
# 4. Placebo Falsification Loop
# ---------------------------------------------------------
print("--- INITIATING PLACEBO FALSIFICATION TESTS ---")
placebo_results = []

# The variables that should NOT be affected by climate policies
placebo_outcomes = [
    #'library_count', 'streetlight_density', 
    'fountain_count', 'bench_count_pc'
]

target_models = ['L1 (Lasso / Logit L1)', 'Causal Forest']

for model_name in target_models:
    ml_dict = models[model_name]
    print(f"\n>> Running Falsification with {model_name}...")
    
    for placebo in placebo_outcomes:
        print(f"  > Testing decoy outcome: {placebo}")
        current_W_cols = [col for col in base_W_cols if col != placebo]
        
        try:
            # 1. Extract Main Point Estimates
            main_ests = fit_and_extract_placebo(ml_dict, df, current_W_cols, placebo)
            
            # 2. Bootstrap Loop for Empirical Standard Errors (PARALLELIZED)
            boot_dists = {p: {'Global_ATE_coef': [], 'Global_ATT_coef': []} for p in regime_mapping.keys()}
            
            # Generate unique random seeds for each iteration
            seeds = np.random.randint(0, 1000000, size=B_BOOTSTRAP)
            
            # joblib.Parallel distributes the iterations across all CPU cores simultaneously
            parallel_results = Parallel(n_jobs=-1, verbose=0)(
                delayed(run_single_bootstrap)(seed, ml_dict, df, current_W_cols, placebo) 
                for seed in seeds
            )
            
            # Unpack the parallel results into the distributions dictionary
            for b_ests in parallel_results:
                if b_ests is not None:
                    for p in regime_mapping.keys():
                        boot_dists[p]['Global_ATE_coef'].append(b_ests[p]['Global_ATE_coef'])
                        boot_dists[p]['Global_ATT_coef'].append(b_ests[p]['Global_ATT_coef'])

            # 3. Calculate p-values and append to final results
            for policy_name in regime_mapping.keys():
                row_data = {'Model': model_name, 'Outcome': placebo, 'Policy': policy_name}
                
                for est_key in ['Global_ATE_coef', 'Global_ATT_coef']:
                    pt_est = main_ests[policy_name][est_key]
                    
                    boot_array = np.array(boot_dists[policy_name][est_key])
                    boot_array = boot_array[~np.isnan(boot_array)]
                    
                    se = np.std(boot_array) if len(boot_array) > 0 else np.nan
                    
                    if not np.isnan(se) and se > 0:
                        z_stat = pt_est / se
                        pval = 2 * (1 - norm.cdf(abs(z_stat)))
                    else:
                        pval = np.nan
                        
                    row_data[est_key] = pt_est
                    row_data[est_key.replace('_coef', '_pval')] = pval
                    
                placebo_results.append(row_data)

        except Exception as e:
            print(f"   [!] Placebo failed for {placebo} using {model_name}: {e}")

# ---------------------------------------------------------
# 5. Export the results
# ---------------------------------------------------------
placebo_df = pd.DataFrame(placebo_results)
placebo_export_path = results_dir / 'sensitivity_placebo_tests2.csv'
placebo_df.to_csv(placebo_export_path, index=False)
print(f"\nSuccess: Falsification estimates safely exported to {placebo_export_path}")