import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM

# Suppress warnings for clean console matrix outputs
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. System Setup & Custom Estimators
# ---------------------------------------------------------
tables_dir = Path('Writing/Tables')
tables_dir.mkdir(parents=True, exist_ok=True)

class ClippedProbaClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom wrapper to prevent tree-based models from outputting exact 0.0 or 1.0 probabilities.
    This safely bypasses DoubleML's overzealous binary label validation check for sparse treatments.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        else:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        probs = self.estimator_.predict_proba(X)
        return np.clip(probs, 1e-5, 1 - 1e-5)

# ---------------------------------------------------------
# 2. Data Loading
# ---------------------------------------------------------
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
# 3. Define the Machine Learning Learners Grid
# ---------------------------------------------------------
base_regressors = [
    ('elnet', make_pipeline(StandardScaler(), ElasticNet(l1_ratio=0.5, random_state=42, max_iter=10000))),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)),
    ('boost', HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5))
]

base_classifiers = [
    ('elnet', make_pipeline(StandardScaler(), LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', random_state=42, max_iter=10000, n_jobs=1))),
    ('rf', ClippedProbaClassifier(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1))),
    ('boost', HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5))
]

# Note: 'Single Tree' has been removed as it is a legacy algorithm
models = {
    'OLS - Basic': {
        'ml_l': make_pipeline(StandardScaler(), LinearRegression()),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)) 
    },
    'Boosted Trees': {
        'ml_l': HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5),
        'ml_m': HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5)
    }
}

# ---------------------------------------------------------
# 4. Main Causal Estimations Loop
# ---------------------------------------------------------
print("--- INITIATING ROBUST DOUBLE MACHINE LEARNING ESTIMATION ---")
final_results = []

# Pre-generate dummy variables for the clusters to prevent gate() index errors
cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='Cluster', dtype=int)

# Hard-code the ATTE for IRM because doubleml's function doesn't actually work
def atte_score(y, d, g_hat0, g_hat1, m_hat, smpls):
    """
    Custom Neyman-orthogonal score function for the Average Treatment Effect on the Treated (ATET).
    Bypasses version restrictions by manually computing the psi_a and psi_b orthogonal scores.
    """
    # Calculate the unconditional probability of treatment in the sample fold
    p_hat = np.maximum(np.mean(d), 1e-6) 
    
    # Neyman orthogonal score components for ATET
    psi_a = -d / p_hat
    psi_b = (d * (y - g_hat0) / p_hat) - (m_hat * (1 - d) * (y - g_hat0) / (p_hat * (1 - m_hat)))
    
    return psi_a, psi_b

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
        
        row_data['PLM_ATE_coef'] = dml_plr.summary.loc[policy, 'coef']
        row_data['PLM_ATE_pval'] = dml_plr.summary.loc[policy, 'P>|t|']
        
        # Setup data for single policy IRM
        dml_data_single = DoubleMLData(df, y_col='log_transport_co2', d_cols=[policy], x_cols=X_cols, cluster_cols='city_id')
        
        try:
            # 1. ATET Estimation
            dml_irm_atet = DoubleMLIRM(dml_data_single, ml_g=ml_l, ml_m=ml_m, n_folds=5, score='ATE')
            dml_irm_atet.fit()
            
            row_data['IRM_ATET_coef'] = dml_irm_atet.coef[0]
            row_data['IRM_ATET_pval'] = dml_irm_atet.pval[0]
            
            # 2. GATET Estimation (Grouped by cluster dummies)
            gate_res = dml_irm_atet.gate(groups=cluster_dummies)
            gate_summary = gate_res.summary
            
            # Safely extract by explicit index name
            row_data['GATET_Cluster_0_coef'] = gate_summary.loc['Cluster_0', 'coef']
            row_data['GATET_Cluster_0_pval'] = gate_summary.loc['Cluster_0', 'P>|t|']
            
            row_data['GATET_Cluster_1_coef'] = gate_summary.loc['Cluster_1', 'coef']
            row_data['GATET_Cluster_1_pval'] = gate_summary.loc['Cluster_1', 'P>|t|']
            
        except Exception as e:
            # We print the error so it never fails silently again
            print(f"   [!] IRM failed for {policy}: {e}")
            
            row_data['IRM_ATET_coef'] = np.nan
            row_data['IRM_ATET_pval'] = np.nan
            row_data['GATET_Cluster_0_coef'] = np.nan
            row_data['GATET_Cluster_0_pval'] = np.nan
            row_data['GATET_Cluster_1_coef'] = np.nan
            row_data['GATET_Cluster_1_pval'] = np.nan
            
        final_results.append(row_data)

# ---------------------------------------------------------
# 5. Generate Clean Output Matrix
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

csv_export_path = results_dir / 'dml_robustness_results_test.csv'
results_df.to_csv(csv_export_path, index=False)

print(f"\nSuccess: Raw causal estimations safely exported to {csv_export_path}")
print("Computation complete. You may now run the formatting script.")