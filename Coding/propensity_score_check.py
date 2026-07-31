import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold

warnings.filterwarnings('ignore')

print("=========================================================")
print(" STANDALONE PROPENSITY SCORE ESTIMATION (ECONML EXACT)   ")
print("=========================================================")

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

print("[1/5] Loading and preparing panel data...")
df = pd.read_csv('Data/urban_emissions_panel_cleaned.csv')

# Reconstruct policy_regime exactly as in the main estimation
# 0 = None, 1 = CP Only, 2 = LEZ Only, 3 = Synergy
df['policy_regime'] = (df['cp_active'] * 1) + (df['lez_active'] * 2)

# Create strict binary indicators for the overlap table to use later
df['is_cp_only'] = (df['policy_regime'] == 1).astype(int)
df['is_lez_only'] = (df['policy_regime'] == 2).astype(int)
df['is_synergy'] = (df['policy_regime'] == 3).astype(int)

# ---------------------------------------------------------
# 2. Covariate Matrix (W) Construction
# ---------------------------------------------------------
print("[2/5] Constructing standardized covariate matrix (W)...")
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

exclude_from_W = [
    # Outcomes, Treatments, IDs
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime', 

    # --- FIXED: Exclude the manually created table indicators! ---
    'is_cp_only', 'is_lez_only', 'is_synergy',
    
    # Post-Treatment Mediators & Feedback Loops (Rely on Mundlak means instead)
    'pm25', 'fleet_diesel_share', 'fleet_electric_share', 'fleet_petrol_share',
    'public_transit_score', 'road_km_pc', 'industry_public',
    'logistics_activity', 'industry_logistics', 'tourism_intensity', # <-- NEW
    'political_green', 'ngo_environment_index', 'electoral_competitiveness' # <-- NEW
]
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]

# ---------------------------------------------------------
# 3. Model Definition (Champion Logit L1)
# ---------------------------------------------------------
print("[3/5] Initializing champion L1 Multinomial Logistic Regression...")
# Plug in the exact value from your optimal_hyperparameters_aipw.txt file
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

champion_pipe = make_pipeline(
    StandardScaler(), 
    LogisticRegression(
        penalty='l1', 
        C=OPTIMAL_C, 
        solver='saga', 
        random_state=42, 
        max_iter=10000, 
        n_jobs=1  # <-- FIXED: Must be 1 to prevent nested thread contention
    )
)

# ---------------------------------------------------------
# 4. Cross-Validated Probability Estimation
# ---------------------------------------------------------
print("[4/5] Calculating cross-validated probabilities...")

# 1. Generate the exact c_dummies used in the main script
c_dummies = pd.get_dummies(df['cluster_id'], prefix='Cluster', dtype=int).to_numpy()
W_mat = df[base_W_cols].to_numpy()

# 2. Concatenate them to perfectly replicate EconML's internal feature space
features_matrix = np.hstack([c_dummies, W_mat])

# 3. Run cross-validation
exact_p_scores_matrix = cross_val_predict(
    champion_pipe,
    features_matrix,                  # <-- FIXED: Now contains [X, W]
    df['policy_regime'].to_numpy(),
    cv=GroupKFold(n_splits=5),
    groups=df['city_id'].to_numpy(),
    method='predict_proba',
    n_jobs=-1                         # Outer parallelization is safe here
)

# ---------------------------------------------------------
# 5. Extract Exact Regime Probabilities & Export
# ---------------------------------------------------------
print("[5/5] Extracting exact discrete regime probabilities and exporting...")

ps_df = pd.DataFrame({
    'city_id': df['city_id'],
    'year': df['year'],
    'policy_regime': df['policy_regime'],
    
    # Binary masks for the Unified Overlap Table
    'is_cp_only': df['is_cp_only'],
    'is_lez_only': df['is_lez_only'],
    'is_synergy': df['is_synergy'],
    
    # Exact probabilities matching EconML's internal AIPW scores
    'propensity_score_cp_only': exact_p_scores_matrix[:, 1],
    'propensity_score_lez_only': exact_p_scores_matrix[:, 2],
    'propensity_score_synergy': exact_p_scores_matrix[:, 3]
})

ps_export_path = results_dir / 'propensity_scores_exact_aipw.csv'
ps_df.to_csv(ps_export_path, index=False)
print(f"\nSuccess: Exact discrete propensity scores saved to {ps_export_path}")
print("=========================================================\n")