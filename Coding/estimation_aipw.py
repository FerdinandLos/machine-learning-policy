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
df['cp_x_lez'] = df['cp_active'] * df['lez_active']
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define Base W (Covariates/Confounders without ANY policies)
exclude_from_W = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez',
    # --- COMPOSITIONAL DUMMY TRAP FIX ---
    # Dropped to serve as the implicit reference categories and prevent rank deficiency
    'industry_public', 'fleet_petrol_share'
]
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]
core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

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
        'max_samples': 0.5,
        # --- SINGULAR MATRIX FIX ---
        # Forces leaves to be large enough to guarantee treatment variance 
        # (preventing local overlap failures for the sparse cp_x_lez policy)
        'min_samples_leaf': 30  
    }

# ---------------------------------------------------------
# 4. Main Causal Estimations Loop (AIPW + Causal Forest)
# ---------------------------------------------------------
print("--- INITIATING AIPW & CAUSAL FOREST ESTIMATION ---")
final_results = []

# Instantiate the GroupKFold cross-validator for panel data
cv_panel = GroupKFold(n_splits=5)

for model_name, ml_dict in models.items():
    print(f"\n>> Estimating with {model_name}...")
    
    for policy in core_policies:
        row_data = {'Model': model_name, 'Policy': policy}
        
        # --- DYNAMIC CO-TREATMENT CONFOUNDING FIX ---
        if policy == 'cp_active':
            dynamic_W_cols = base_W_cols + ['lez_active']
        elif policy == 'lez_active':
            dynamic_W_cols = base_W_cols + ['cp_active']
        elif policy == 'cp_x_lez':
            # Include both standalone main effects to isolate the true synergy
            dynamic_W_cols = base_W_cols + ['cp_active', 'lez_active']
            
        W_matrix = df[dynamic_W_cols].to_numpy()
        city_groups = df['city_id'].to_numpy()
        
        try:
            # Instantiate Estimator
            if ml_dict['type'] == 'dr':
                estimator = LinearDRLearner(
                    model_regression=clone(ml_dict['ml_l']),
                    model_propensity=clone(ml_dict['ml_m']),
                    min_propensity=0.01, # Trim extreme propensity weights
                    cv=cv_panel,
                    random_state=42,
                    fit_cate_intercept=False
                )
            elif ml_dict['type'] == 'causal_forest':
                estimator = CausalForestDML(
                    n_estimators=ml_dict['n_estimators'],
                    max_depth=ml_dict['max_depth'],
                    max_samples=ml_dict['max_samples'],
                    cv=cv_panel,
                    random_state=42,
                    fit_intercept=False
                )

            # --- RUNTIME FIX: Single Fit ---
            # Fitting once with X=cluster_dummies allows us to extract both global and subgroup ATEs
            estimator.fit(
                Y=df['log_transport_co2'].to_numpy(),
                T=df[policy].to_numpy(),
                X=cluster_dummies_array,
                W=W_matrix,
                groups=city_groups
            )

            # 1. Global ATE 
            # We explicitly pass the full X matrix so EconML can average the CATEs across the entire sample
            row_data['Global_ATE_coef'] = np.atleast_1d(estimator.ate(X=cluster_dummies_array))[0]
            row_data['Global_ATE_pval'] = np.atleast_1d(estimator.ate_inference(X=cluster_dummies_array).pvalue())[0]

            # 2. ATET across adopters
            treated_mask = (df[policy].to_numpy() == 1)
            X_treated = cluster_dummies_array[treated_mask]
            
            row_data['CATE_ATET_coef'] = np.atleast_1d(estimator.ate(X=X_treated))[0]
            row_data['CATE_ATET_pval'] = np.atleast_1d(estimator.ate_inference(X=X_treated).pvalue())[0]

            # 3. Intra-Cluster CATEs (Non-Parametric extraction)
            X_cluster_0 = cluster_dummies_array[cluster_dummies['Cluster_0'].to_numpy() == 1]
            X_cluster_1 = cluster_dummies_array[cluster_dummies['Cluster_1'].to_numpy() == 1]

            row_data['CATE_Cluster_0_coef'] = np.atleast_1d(estimator.ate(X=X_cluster_0))[0]
            row_data['CATE_Cluster_0_pval'] = np.atleast_1d(estimator.ate_inference(X=X_cluster_0).pvalue())[0]
            
            row_data['CATE_Cluster_1_coef'] = np.atleast_1d(estimator.ate(X=X_cluster_1))[0]
            row_data['CATE_Cluster_1_pval'] = np.atleast_1d(estimator.ate_inference(X=X_cluster_1).pvalue())[0]

        except Exception as e:
            print(f"   [!] Estimation failed for {policy} with {model_name}: {e}")
            row_data['Global_ATE_coef'] = np.nan
            row_data['Global_ATE_pval'] = np.nan
            row_data['CATE_ATET_coef'] = np.nan
            row_data['CATE_ATET_pval'] = np.nan
            row_data['CATE_Cluster_0_coef'] = np.nan
            row_data['CATE_Cluster_0_pval'] = np.nan
            row_data['CATE_Cluster_1_coef'] = np.nan
            row_data['CATE_Cluster_1_pval'] = np.nan

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

# Figure 3 plots cp_active, so we must include lez_active in W to remain honest
fig3_W_cols = base_W_cols + ['lez_active']

exact_p_scores = cross_val_predict(
    champion_clf,
    df[fig3_W_cols].to_numpy(),
    df['cp_active'].to_numpy(),
    cv=GroupKFold(n_splits=5),
    groups=df['city_id'].to_numpy(),
    method='predict_proba',
    n_jobs=-1
)[:, 1]

ps_df = pd.DataFrame({
    'cp_active': df['cp_active'],
    'propensity_score': exact_p_scores
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
    f.write("Grid Search Options: [0.1, 0.5, 0.9, 0.99]\n")
    
    for policy in core_policies:
        f.write(f"\n--- POLICY: {policy.upper()} ---\n")
        
        # Re-create the exact dynamic control matrix for this policy
        if policy == 'cp_active':
            dyn_w = base_W_cols + ['lez_active']
        elif policy == 'lez_active':
            dyn_w = base_W_cols + ['cp_active']
        elif policy == 'cp_x_lez':
            dyn_w = base_W_cols + ['cp_active', 'lez_active']
            
        # Outcome Model Hyperparameters
        elnet_y_pipe = clone(models['Elastic Net']['ml_l'])
        elnet_y_pipe.fit(df[dyn_w].to_numpy(), df['log_transport_co2'].to_numpy())
        best_l1_y = elnet_y_pipe.steps[1][1].l1_ratio_
        f.write(f"Outcome Model (Log Transport CO2) Selected l1_ratio: {best_l1_y}\n")
        
        # Treatment Model Hyperparameters
        elnet_t_pipe = clone(models['Elastic Net']['ml_m'])
        elnet_t_pipe.fit(df[dyn_w].to_numpy(), df[policy].to_numpy())
        best_l1_t = elnet_t_pipe.steps[1][1].l1_ratio_[0]
        f.write(f"Treatment Model ({policy.upper()}) Selected l1_ratio: {best_l1_t}\n")

print(f"Success: Optimal hyperparameters saved to {hyperparam_path}")