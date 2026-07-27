import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, ElasticNet, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_predict
from econml.dml import LinearDML

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
    Ensures safe propensity score clipping for rare events (like cp_x_lez).
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

# Define X (Covariates/Confounders)
exclude_from_X = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez'
]
X_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_X]
core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

# Pre-generate dummy variables for the clusters for GATE estimation
cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='Cluster', dtype=int)

# ---------------------------------------------------------
# 3. Define the Machine Learning Learners Grid (Test Scope)
# ---------------------------------------------------------
models = {
    'Elastic Net': {
        'ml_l': make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9, 0.99], random_state=42, max_iter=10000, n_jobs=-1)),
        # Explicitly added penalty='elasticnet' so it actually uses the l1_ratios
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9, 0.99], solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    }
}

# ---------------------------------------------------------
# 4. Main Causal Estimations Loop (EconML)
# ---------------------------------------------------------
print("--- INITIATING ROBUST ECON-ML ESTIMATION ---")
final_results = []

for model_name, ml_dict in models.items():
    print(f"\n>> Estimating with {model_name}...")
    ml_l = ml_dict['ml_l']
    ml_m = ml_dict['ml_m']
    
    # A. PLM (Baseline Average Effects - all policies at once)
    # Wrap regressor to handle predicting multiple treatments simultaneously
    ml_t_multi = MultiOutputRegressor(clone(ml_l))
    
    # discrete_treatment=False treats policies as continuous components to partial out covariates
    dml_plm = LinearDML(model_y=clone(ml_l), model_t=ml_t_multi, discrete_treatment=False, cv=5, random_state=42)
    dml_plm.fit(Y=df['log_transport_co2'], T=df[core_policies], X=None, W=df[X_cols])
    
    # Extract PLM coefficients and p-values
    plm_coefs = dml_plm.const_marginal_effect().flatten()
    plm_pvals = dml_plm.const_marginal_effect_inference().pvalue().flatten()
    plm_results = {pol: {'coef': plm_coefs[i], 'pval': plm_pvals[i]} for i, pol in enumerate(core_policies)}
    
    # B. IRM ATE & GATE (Policy-by-Policy)
    for policy in core_policies:
        row_data = {'Model': model_name, 'Policy': policy}
        row_data['PLM_ATE_coef'] = plm_results[policy]['coef']
        row_data['PLM_ATE_pval'] = plm_results[policy]['pval']
        
        try:
            # fit_cate_intercept=False projects the CATE directly onto the cluster dummies
            dml_irm = LinearDML(model_y=clone(ml_l), model_t=clone(ml_m), discrete_treatment=True, cv=5, random_state=42, fit_cate_intercept=False)
            dml_irm.fit(Y=df['log_transport_co2'], T=df[policy], X=cluster_dummies, W=df[X_cols])
            
            # --- THE ATET FIX ---
            # Filter the cluster dummies to ONLY include cities where the policy is active
            treated_mask = (df[policy] == 1)
            X_treated = cluster_dummies[treated_mask]
            
            # 1. True ATET Estimation (Average effect only across the treated cities)
            row_data['IRM_ATET_coef'] = dml_irm.ate(X=X_treated)
            row_data['IRM_ATET_pval'] = dml_irm.ate_inference(X=X_treated).pvalue()
            
            # 2. GATET Estimation (Heterogeneity by Cluster)
            # Because we project directly onto the clusters, the coefficients are the intra-cluster effects
            gatet_coefs = dml_irm.coef_.flatten()
            gatet_pvals = dml_irm.coef__inference().pvalue().flatten()
            
            row_data['GATET_Cluster_0_coef'] = gatet_coefs[0]
            row_data['GATET_Cluster_0_pval'] = gatet_pvals[0]
            
            row_data['GATET_Cluster_1_coef'] = gatet_coefs[1]
            row_data['GATET_Cluster_1_pval'] = gatet_pvals[1]
            
        except Exception as e:
            print(f"   [!] IRM failed for {policy}: {e}")
            row_data['IRM_ATET_coef'] = np.nan
            row_data['IRM_ATET_pval'] = np.nan
            row_data['GATET_Cluster_0_coef'] = np.nan
            row_data['GATET_Cluster_0_pval'] = np.nan
            row_data['GATET_Cluster_1_coef'] = np.nan
            row_data['GATET_Cluster_1_pval'] = np.nan
            
        final_results.append(row_data)


# ---------------------------------------------------------
# Extracting Optimal Hyperparameters for Thesis Text
# ---------------------------------------------------------
results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)

print("\n--- EXTRACTING OPTIMAL HYPERPARAMETERS ---")
# Clone the pipelines so we don't accidentally overwrite anything
elnet_y_pipe = clone(models['Elastic Net']['ml_l'])
elnet_t_pipe = clone(models['Elastic Net']['ml_m'])

# 1. Fit Outcome Model (CO2) on the full dataset
elnet_y_pipe.fit(df[X_cols], df['log_transport_co2'])
# .steps[1][1] accesses the ElasticNetCV step inside the pipeline
best_l1_y = elnet_y_pipe.steps[1][1].l1_ratio_

# 2. Fit Treatment Model (Propensity for CP) on the full dataset
elnet_t_pipe.fit(df[X_cols], df['cp_active'])
# LogisticRegressionCV returns an array, so we extract the first item [0]
best_l1_t = elnet_t_pipe.steps[1][1].l1_ratio_[0]

# Save to a text file for easy reference while writing
hyperparam_path = results_dir / 'optimal_hyperparameters.txt'
with open(hyperparam_path, 'w') as f:
    f.write("--- ELASTIC NET OPTIMAL HYPERPARAMETERS (FULL SAMPLE) ---\n")
    f.write("Grid Search Options: [0.1, 0.5, 0.9, 0.99]\n\n")
    f.write(f"Outcome Model (Log Transport CO2) Selected l1_ratio: {best_l1_y}\n")
    f.write(f"Treatment Model (CP Active) Selected l1_ratio: {best_l1_t}\n")

print(f"Success: Optimal hyperparameters saved to {hyperparam_path}")