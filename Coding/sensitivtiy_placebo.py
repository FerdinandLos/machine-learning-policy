import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

# EconML Imports for AIPW Doubly Robust Estimation
from econml.dr import LinearDRLearner
from econml.dml import CausalForestDML

# Suppress warnings for clean console matrix outputs
warnings.filterwarnings('ignore')

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
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime',
    # --- COMPOSITIONAL DUMMY TRAP FIX ---
    'industry_public', 'fleet_petrol_share'
]
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]
core_policies = ['cp_active', 'lez_active', 'policy_regime']

# Pre-generate dummy variables for the clusters for CATE estimation
cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='Cluster', dtype=int)
cluster_dummies_array = cluster_dummies.to_numpy()

# ---------------------------------------------------------
# 3. Define the Learners Grid
# ---------------------------------------------------------
models = {
    'L1 (Lasso / Logit L1)': {
        'type': 'dr',
        'ml_l': make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l1', solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    },
    'Causal Forest': {
        'type': 'causal_forest',
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_leaf': 50  
    }
}

# ---------------------------------------------------------
# 4. Placebo Falsification Loop
# ---------------------------------------------------------
print("--- INITIATING PLACEBO FALSIFICATION TESTS ---")
placebo_results = []
cv_panel = GroupKFold(n_splits=5)

regime_mapping = {
    'cp_active': 1,
    'lez_active': 2,
    'cp_x_lez': 3
}

# The variables that should NOT be affected by climate policies
placebo_outcomes = [
    'library_count', 'streetlight_density', 
    'fountain_count', 'bench_count_pc', 'flagpole_count', 'sister_city_count'
]

target_models = ['L1 (Lasso / Logit L1)', 'Causal Forest']

for model_name in target_models:
    ml_dict = models[model_name]
    print(f"\n>> Running Falsification with {model_name}...")
    
    for placebo in placebo_outcomes:
        print(f"  > Testing decoy outcome: {placebo}")
        
        # CRITICAL FIX: Remove the placebo outcome from the W control matrix
        current_W_cols = [col for col in base_W_cols if col != placebo]
        W_matrix = df[current_W_cols].to_numpy()
        
        Y_arr = df[placebo].to_numpy()
        T_arr = df['policy_regime'].to_numpy()
        city_groups = df['city_id'].to_numpy()
        
        try:
            # Dynamically instantiate based on algorithm type
            if ml_dict['type'] == 'dr':
                estimator = LinearDRLearner(
                    model_regression=clone(ml_dict['ml_l']),
                    model_propensity=clone(ml_dict['ml_m']),
                    min_propensity=0.01, 
                    cv=cv_panel,
                    random_state=42
                )
            elif ml_dict['type'] == 'causal_forest':
                estimator = CausalForestDML(
                    n_estimators=ml_dict['n_estimators'],
                    max_depth=ml_dict['max_depth'],
                    min_samples_leaf=ml_dict['min_samples_leaf'],
                    discrete_treatment=True,
                    cv=cv_panel,
                    random_state=42
                )

            estimator.fit(
                Y=Y_arr,
                T=T_arr,
                X=cluster_dummies_array,
                W=W_matrix,
                groups=city_groups
            )

            for policy_name, t_val in regime_mapping.items():
                # ADDED: Store the Model name for the LaTeX pivoting script
                row_data = {'Model': model_name, 'Outcome': placebo, 'Policy': policy_name}
                
                # Global ATE
                row_data['Global_ATE_coef'] = np.atleast_1d(estimator.ate(X=cluster_dummies_array, T0=0, T1=t_val))[0]
                row_data['Global_ATE_pval'] = np.atleast_1d(estimator.ate_inference(X=cluster_dummies_array, T0=0, T1=t_val).pvalue())[0]

                # Global ATT
                treated_mask = (df['policy_regime'].to_numpy() == t_val)
                X_treated = cluster_dummies_array[treated_mask]
                row_data['Global_ATT_coef'] = np.atleast_1d(estimator.ate(X=X_treated, T0=0, T1=t_val))[0]
                row_data['Global_ATT_pval'] = np.atleast_1d(estimator.ate_inference(X=X_treated, T0=0, T1=t_val).pvalue())[0]
                
                placebo_results.append(row_data)

        except Exception as e:
            print(f"   [!] Placebo failed for {placebo} using {model_name}: {e}")

# Export the results
placebo_df = pd.DataFrame(placebo_results)
placebo_export_path = results_dir / 'sensitivity_placebo_tests.csv'
placebo_df.to_csv(placebo_export_path, index=False)
print(f"\nSuccess: Falsification estimates safely exported to {placebo_export_path}")