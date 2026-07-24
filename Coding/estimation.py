import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, ElasticNet, LogisticRegression
from sklearn.pipeline import make_pipeline
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM

# Suppress warnings for clean console matrix outputs
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
tables_dir = Path('Writing/Tables')
tables_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('Data/urban_emissions_panel_cleaned.csv')
df['cp_x_lez'] = df['cp_active'] * df['lez_active']
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define X (Covariates)
exclude_from_X = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez'
]
X_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_X]
core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

# ---------------------------------------------------------
# 2. Define the Machine Learning Learners Grid
# ---------------------------------------------------------
# Base estimators for Stacking MUST be static to provide a consistent baseline
base_regressors = [
    ('elnet', make_pipeline(StandardScaler(), ElasticNet(l1_ratio=0.5, random_state=42, max_iter=10000))),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)),
    ('boost', HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5))
]

base_classifiers = [
    ('elnet', make_pipeline(StandardScaler(), LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', random_state=42, max_iter=10000, n_jobs=1))),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)),
    ('boost', HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5))
]

# Standalone models explicitly tuned with cv=5 and max_iter=10000 per your rigor requirements
models = {
    'OLS - Basic': {
        'ml_l': make_pipeline(StandardScaler(), LinearRegression()),
        # Must be an unpenalized probability model (predict_proba) for DoubleMLIRM
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)) 
    },
    'L1 (Lasso / Logit L1)': {
        'ml_l': make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, l1_ratios=[1.0], solver='saga', scoring='neg_log_loss', use_legacy_attributes=False, random_state=42, max_iter=10000, n_jobs=1))
    },
    'L2 (Ridge / Logit L2)': {
        'ml_l': make_pipeline(StandardScaler(), RidgeCV(cv=5)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, l1_ratios=[0.0], solver='saga', scoring='neg_log_loss', use_legacy_attributes=False, random_state=42, max_iter=10000, n_jobs=1))
    },
    'Elastic Net': {
        'ml_l': make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], random_state=42, max_iter=10000, n_jobs=1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, l1_ratios=[0.1, 0.5, 0.9], solver='saga', scoring='neg_log_loss', use_legacy_attributes=False, random_state=42, max_iter=10000, n_jobs=1))
    },
    'Random Forest': {
        'ml_l': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1),
        'ml_m': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
    },
    'Boosted Trees': {
        'ml_l': HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5),
        'ml_m': HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5)
    },
    'Ensemble (Stacking)': {
        'ml_l': StackingRegressor(estimators=base_regressors, final_estimator=make_pipeline(StandardScaler(), RidgeCV(cv=5))),
        'ml_m': StackingClassifier(estimators=base_classifiers, final_estimator=make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, l1_ratios=[0.0], solver='saga', scoring='neg_log_loss', use_legacy_attributes=False, max_iter=10000)))
    }
}

# ---------------------------------------------------------
# 3. Main Causal Estimations Loop
# ---------------------------------------------------------
print("--- INITIATING ROBUST DOUBLE MACHINE LEARNING ESTIMATION ---")
final_results = []

for model_name, ml_dict in models.items():
    print(f"\n>> Estimating with {model_name}...")
    ml_l = ml_dict['ml_l']
    ml_m = ml_dict['ml_m']
    
    # A. PLM (Baseline Average Effects - all policies at once)
    dml_data_plm = DoubleMLData(df, y_col='log_transport_co2', d_cols=core_policies, x_cols=X_cols, cluster_cols='city_id')
    dml_plr = DoubleMLPLR(dml_data_plm, ml_l=ml_l, ml_m=ml_m, n_folds=5)
    dml_plr.fit()
    
    # B. IRM ATET & GATET (Policy-by-Policy)
    for policy in core_policies:
        row_data = {'Model': model_name, 'Policy': policy}
        
        # FIX: Query the summary DataFrame by string index instead of indexing the NumPy array
        row_data['PLM_ATE_coef'] = dml_plr.summary.loc[policy, 'coef']
        row_data['PLM_ATE_pval'] = dml_plr.summary.loc[policy, 'P>|t|']
        
        # Setup data for single policy IRM
        dml_data_single = DoubleMLData(df, y_col='log_transport_co2', d_cols=[policy], x_cols=X_cols, cluster_cols='city_id')
        
        try:
            # 1. ATET Estimation
            dml_irm_atet = DoubleMLIRM(dml_data_single, ml_g=ml_l, ml_m=ml_m, n_folds=5, score='ATTE')
            dml_irm_atet.fit()
            
            # Extract ATET (since there is only 1 treatment here, [0] works safely for the NumPy array)
            row_data['IRM_ATET_coef'] = dml_irm_atet.coef[0]
            row_data['IRM_ATET_pval'] = dml_irm_atet.pval[0]
            
            # 2. GATET Estimation (Grouped by cluster_id)
            gate_res = dml_irm_atet.gate(groups=df[['cluster_id']])
            gate_summary = gate_res.summary
            
            # Extract raw coefficients and p-values for cluster 0 and cluster 1
            row_data['GATET_Cluster_0_coef'] = gate_summary.iloc[0]['coef']
            row_data['GATET_Cluster_0_pval'] = gate_summary.iloc[0]['P>|t|']
            
            row_data['GATET_Cluster_1_coef'] = gate_summary.iloc[1]['coef']
            row_data['GATET_Cluster_1_pval'] = gate_summary.iloc[1]['P>|t|']
            
        except Exception as e:
            # Catch overlap violations and output NaN (Not a Number)
            row_data['IRM_ATET_coef'] = np.nan
            row_data['IRM_ATET_pval'] = np.nan
            
            row_data['GATET_Cluster_0_coef'] = np.nan
            row_data['GATET_Cluster_0_pval'] = np.nan
            row_data['GATET_Cluster_1_coef'] = np.nan
            row_data['GATET_Cluster_1_pval'] = np.nan
            
        final_results.append(row_data)

# ---------------------------------------------------------
# 4. Generate Clean Output Matrix
# ---------------------------------------------------------
results_df = pd.DataFrame(final_results)

print("\n--- FINAL CAUSAL ESTIMATES ACROSS ALL ML ALGORITHMS ---")
for policy in core_policies:
    print(f"\n--- POLICY: {policy.upper()} ---")
    policy_df = results_df[results_df['Policy'] == policy].drop(columns=['Policy']).set_index('Model')
    print(policy_df.to_string())

# Create a dedicated directory for raw pipeline outputs if it doesn't exist
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

csv_export_path = results_dir / 'dml_robustness_results.csv'
results_df.to_csv(csv_export_path, index=False)

print(f"\nSuccess: Raw causal estimations safely exported to {csv_export_path}")
print("Computation complete. You may now run the formatting script.")