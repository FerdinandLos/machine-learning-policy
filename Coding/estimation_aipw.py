import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, GroupKFold

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
# 0 = No Policies
# 1 = CP Only (1 * 1 + 0 * 2)
# 2 = LEZ Only (0 * 1 + 1 * 2)
# 3 = Both / Synergy (1 * 1 + 1 * 2)
df['policy_regime'] = (df['cp_active'] * 1) + (df['lez_active'] * 2)

# Run this quick sanity check in your console
print(pd.crosstab(df['policy_regime'], df['cluster_id'], margins=True))

year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define Base W (Covariates/Confounders without ANY policies)
exclude_from_W = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime',
    # --- COMPOSITIONAL DUMMY TRAP FIX ---
    # Dropped to serve as the implicit reference categories and prevent rank deficiency
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
# Linear AIPW Models + Causal Forest
models = {
    'OLS - Basic': {
        'type': 'dr',
        'ml_l': make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, n_jobs=-1)) 
    },
    'L1 (Lasso / Logit L1)': {
        'type': 'dr',
        'ml_l': make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l1', solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    },
    'L2 (Ridge / Logit L2)': {
        'type': 'dr',
        'ml_l': make_pipeline(StandardScaler(), RidgeCV(cv=5)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l2', solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    },
    'Elastic Net': {
        'type': 'dr',
        'ml_l': make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9, 0.99], random_state=42, max_iter=10000, n_jobs=-1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9, 0.99], solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    },
    'Causal Forest': {
        'type': 'causal_forest',
        'n_estimators': 200,
        'max_depth': 5,
        # --- SINGULAR MATRIX FIX ---
        'min_samples_leaf': 50  
    }
}

# ---------------------------------------------------------
# 4. Main Causal Estimations Loop (Categorical AIPW + Causal Forest)
# ---------------------------------------------------------
print("--- INITIATING AIPW & CAUSAL FOREST ESTIMATION ---")
final_results = []

# Instantiate the GroupKFold cross-validator for panel data
cv_panel = GroupKFold(n_splits=5)

# Map the regimes back to their policy names for the results dataframe
regime_mapping = {
    'cp_active': 1,  # T=1
    'lez_active': 2, # T=2
    'cp_x_lez': 3    # T=3
}

for model_name, ml_dict in models.items():
    print(f"\n>> Estimating with {model_name}...")
    
    # 1. Prepare the static matrices
    W_matrix = df[base_W_cols].to_numpy()
    T_arr = df['policy_regime'].to_numpy() # The 0, 1, 2, 3 categorical array
    Y_arr = df['log_transport_co2'].to_numpy()
    city_groups = df['city_id'].to_numpy()
    
    try:
        # Instantiate Estimator
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

        # Fit ONCE per algorithm over the entire categorical regime
        estimator.fit(
            Y=Y_arr,
            T=T_arr,
            X=cluster_dummies_array,
            W=W_matrix,
            groups=city_groups
        )

        # 2. Extract the ATE, Global ATT, GATE, and GATT for each policy contrast
        for policy_name, t_val in regime_mapping.items():
            row_data = {'Model': model_name, 'Policy': policy_name}
            
            # --- 2A. Global ATE ---
            row_data['Global_ATE_coef'] = np.atleast_1d(estimator.ate(X=cluster_dummies_array, T0=0, T1=t_val))[0]
            row_data['Global_ATE_pval'] = np.atleast_1d(estimator.ate_inference(X=cluster_dummies_array, T0=0, T1=t_val).pvalue())[0]

            # --- 2B. Global ATT (across all true adopters) ---
            treated_mask = (df['policy_regime'].to_numpy() == t_val)
            X_treated = cluster_dummies_array[treated_mask]
            
            row_data['Global_ATT_coef'] = np.atleast_1d(estimator.ate(X=X_treated, T0=0, T1=t_val))[0]
            row_data['Global_ATT_pval'] = np.atleast_1d(estimator.ate_inference(X=X_treated, T0=0, T1=t_val).pvalue())[0]

            # Prepare Cluster Masks
            cluster_0_mask = (df['cluster_id'].to_numpy() == 0)
            cluster_1_mask = (df['cluster_id'].to_numpy() == 1)

            # --- 2C. Intra-Cluster GATEs (Average across ALL cities in the cluster) ---
            X_cluster_0 = cluster_dummies_array[cluster_0_mask]
            X_cluster_1 = cluster_dummies_array[cluster_1_mask]

            row_data['GATE_Cluster_0_coef'] = np.atleast_1d(estimator.ate(X=X_cluster_0, T0=0, T1=t_val))[0]
            row_data['GATE_Cluster_0_pval'] = np.atleast_1d(estimator.ate_inference(X=X_cluster_0, T0=0, T1=t_val).pvalue())[0]
            
            row_data['GATE_Cluster_1_coef'] = np.atleast_1d(estimator.ate(X=X_cluster_1, T0=0, T1=t_val))[0]
            row_data['GATE_Cluster_1_pval'] = np.atleast_1d(estimator.ate_inference(X=X_cluster_1, T0=0, T1=t_val).pvalue())[0]

            # --- 2D. Intra-Cluster GATTs (Average strictly across TREATED cities in the cluster) ---
            X_treated_cluster_0 = cluster_dummies_array[treated_mask & cluster_0_mask]
            X_treated_cluster_1 = cluster_dummies_array[treated_mask & cluster_1_mask]

            row_data['GATT_Cluster_0_coef'] = np.atleast_1d(estimator.ate(X=X_treated_cluster_0, T0=0, T1=t_val))[0]
            row_data['GATT_Cluster_0_pval'] = np.atleast_1d(estimator.ate_inference(X=X_treated_cluster_0, T0=0, T1=t_val).pvalue())[0]
            
            row_data['GATT_Cluster_1_coef'] = np.atleast_1d(estimator.ate(X=X_treated_cluster_1, T0=0, T1=t_val))[0]
            row_data['GATT_Cluster_1_pval'] = np.atleast_1d(estimator.ate_inference(X=X_treated_cluster_1, T0=0, T1=t_val).pvalue())[0]
            
            final_results.append(row_data)

    except Exception as e:
        print(f"   [!] Estimation failed for {model_name}: {e}")
        # Append empty rows containing the new column names if the model fails
        for policy_name in regime_mapping.keys():
            row_data = {'Model': model_name, 'Policy': policy_name}
            for col in ['Global_ATE_coef', 'Global_ATE_pval', 'Global_ATT_coef', 'Global_ATT_pval', 
                        'GATE_Cluster_0_coef', 'GATE_Cluster_0_pval', 'GATE_Cluster_1_coef', 'GATE_Cluster_1_pval',
                        'GATT_Cluster_0_coef', 'GATT_Cluster_0_pval', 'GATT_Cluster_1_coef', 'GATT_Cluster_1_pval']:
                row_data[col] = np.nan
            final_results.append(row_data)

# ---------------------------------------------------------
# 5. Generate Clean Output Matrix
# ---------------------------------------------------------
results_df = pd.DataFrame(final_results)

print("\n--- FINAL CAUSAL ESTIMATES (AIPW & CAUSAL FOREST) ---")
for policy in core_policies:
    print(f"\n--- POLICY: {policy.upper()} ---")
    policy_df = results_df[results_df['Policy'] == policy].drop(columns=['Policy']).set_index('Model')
    print(policy_df.to_string())

csv_export_path = results_dir / 'dml_robustness_results_aipw.csv'
results_df.to_csv(csv_export_path, index=False)

print(f"\nSuccess: Raw causal estimations safely exported to {csv_export_path}")

# ---------------------------------------------------------
# 6. Extract Propensity Scores (Champion Model)
# ---------------------------------------------------------
print("\n--- EXPORTING EXACT PROPENSITY SCORES FOR FIGURE 3 ---")
champion_clf = clone(models['L1 (Lasso / Logit L1)']['ml_m'])

# Scikit-learn handles multiclass LogisticRegression natively
exact_p_scores_matrix = cross_val_predict(
    champion_clf,
    df[base_W_cols].to_numpy(),
    df['policy_regime'].to_numpy(),
    cv=GroupKFold(n_splits=5),
    groups=df['city_id'].to_numpy(),
    method='predict_proba',
    n_jobs=-1
)

# Extract the probability of Class 1 (cp_active) 
ps_df = pd.DataFrame({
    'cp_active': df['cp_active'],
    'propensity_score': exact_p_scores_matrix[:, 1]
})

ps_export_path = results_dir / 'propensity_scores_exact_aipw.csv'
ps_df.to_csv(ps_export_path, index=False)
print(f"Success: Exact propensity scores saved to {ps_export_path}")

# ---------------------------------------------------------
# 7. Extracting Optimal Hyperparameters for Thesis Text
# ---------------------------------------------------------
print("\n--- EXTRACTING OPTIMAL HYPERPARAMETERS ---")
hyperparam_path = results_dir / 'optimal_hyperparameters_aipw.txt'

with open(hyperparam_path, 'w') as f:
    f.write("--- ELASTIC NET OPTIMAL HYPERPARAMETERS (FULL SAMPLE) ---\n")
    f.write("Grid Search Options: [0.1, 0.5, 0.9, 0.99]\n\n")
        
    # Outcome Model Hyperparameters
    elnet_y_pipe = clone(models['Elastic Net']['ml_l'])
    elnet_y_pipe.fit(df[base_W_cols].to_numpy(), df['log_transport_co2'].to_numpy())
    best_l1_y = elnet_y_pipe.steps[1][1].l1_ratio_
    f.write(f"Outcome Model (Log Transport CO2) Selected l1_ratio: {best_l1_y}\n")
    
    # Treatment Model Hyperparameters (Multinomial)
    elnet_t_pipe = clone(models['Elastic Net']['ml_m'])
    elnet_t_pipe.fit(df[base_W_cols].to_numpy(), df['policy_regime'].to_numpy())
    best_l1_t = elnet_t_pipe.steps[1][1].l1_ratio_[0] 
    f.write(f"Treatment Model (Categorical Policy Regime) Selected l1_ratio: {best_l1_t}\n")

print(f"Success: Optimal hyperparameters saved to {hyperparam_path}")