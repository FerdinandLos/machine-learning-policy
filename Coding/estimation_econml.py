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

# Pre-generate dummy variables for the clusters for CATE estimation
cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='Cluster', dtype=int)

# ---------------------------------------------------------
# 3. Define the Machine Learning Learners Grid 
# ---------------------------------------------------------
models = {
    'OLS - Basic': {
        'ml_l': make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, n_jobs=-1)) 
    },
    'L1 (Lasso / Logit L1)': {
        'ml_l': make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l1', solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    },
    'L2 (Ridge / Logit L2)': {
        'ml_l': make_pipeline(StandardScaler(), RidgeCV(cv=5)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l2', solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    },
    'Elastic Net': {
        'ml_l': make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9, 0.99], random_state=42, max_iter=10000, n_jobs=-1)),
        'ml_m': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9, 0.99], solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))
    },
    'Random Forest': {
        'ml_l': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'ml_m': ClippedProbaClassifier(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1))
    },
    'Boosted Trees': {
        'ml_l': HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5),
        'ml_m': ClippedProbaClassifier(HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5))
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
    
    # A. Constant Partially Linear Model (Baseline Global ATE)
    ml_t_multi = MultiOutputRegressor(clone(ml_l))
    
    dml_global = LinearDML(model_y=clone(ml_l), model_t=ml_t_multi, discrete_treatment=False, cv=5, random_state=42)
    dml_global.fit(Y=df['log_transport_co2'], T=df[core_policies], X=None, W=df[X_cols])
    
    global_coefs = dml_global.const_marginal_effect().flatten()
    global_pvals = dml_global.const_marginal_effect_inference().pvalue().flatten()
    global_results = {pol: {'coef': global_coefs[i], 'pval': global_pvals[i]} for i, pol in enumerate(core_policies)}
    
    # B. Heterogeneous Partially Linear Model (CATE Estimation)
    for policy in core_policies:
        row_data = {'Model': model_name, 'Policy': policy}
        row_data['Global_ATE_coef'] = global_results[policy]['coef']
        row_data['Global_ATE_pval'] = global_results[policy]['pval']
        
        try:
            dml_cate = LinearDML(model_y=clone(ml_l), model_t=clone(ml_m), discrete_treatment=True, cv=5, random_state=42, fit_cate_intercept=False)
            dml_cate.fit(Y=df['log_transport_co2'], T=df[policy], X=cluster_dummies, W=df[X_cols])
            
            # --- THE ATET FIX ---
            treated_mask = (df[policy] == 1)
            X_treated = cluster_dummies[treated_mask]
            
            # 1. True ATET Estimation (Average effect only across the treated cities)
            row_data['CATE_ATET_coef'] = dml_cate.ate(X=X_treated)
            row_data['CATE_ATET_pval'] = dml_cate.ate_inference(X=X_treated).pvalue()
            
            # 2. Intra-Cluster CATE Estimation 
            cate_coefs = dml_cate.coef_.flatten()
            cate_pvals = dml_cate.coef__inference().pvalue().flatten()
            
            row_data['CATE_Cluster_0_coef'] = cate_coefs[0]
            row_data['CATE_Cluster_0_pval'] = cate_pvals[0]
            
            row_data['CATE_Cluster_1_coef'] = cate_coefs[1]
            row_data['CATE_Cluster_1_pval'] = cate_pvals[1]
            
        except Exception as e:
            print(f"   [!] CATE failed for {policy}: {e}")
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

print("\n--- FINAL CAUSAL ESTIMATES (TEST SCOPE) ---")
for policy in core_policies:
    print(f"\n--- POLICY: {policy.upper()} ---")
    policy_df = results_df[results_df['Policy'] == policy].drop(columns=['Policy']).set_index('Model')
    print(policy_df.to_string())

results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)
csv_export_path = results_dir / 'dml_robustness_results_econml.csv'
results_df.to_csv(csv_export_path, index=False)

print(f"\nSuccess: Raw causal estimations safely exported to {csv_export_path}")

# ---------------------------------------------------------
# 6. Extract Propensity Scores (Champion Model)
# ---------------------------------------------------------
print("\n--- EXPORTING EXACT PROPENSITY SCORES FOR FIGURE 3 ---")
champion_clf = models['L1 (Lasso / Logit L1)']['ml_m']

exact_p_scores = cross_val_predict(
    champion_clf, 
    df[X_cols], 
    df['cp_active'], 
    cv=5, 
    method='predict_proba', 
    n_jobs=-1
)[:, 1]

ps_df = pd.DataFrame({
    'cp_active': df['cp_active'],
    'propensity_score': exact_p_scores
})

ps_export_path = results_dir / 'propensity_scores_exact.csv'
ps_df.to_csv(ps_export_path, index=False)
print(f"Success: Exact propensity scores saved to {ps_export_path}")

# ---------------------------------------------------------
# 7. Extracting Optimal Hyperparameters for Thesis Text
# ---------------------------------------------------------
print("\n--- EXTRACTING OPTIMAL HYPERPARAMETERS ---")
elnet_y_pipe = clone(models['Elastic Net']['ml_l'])
elnet_t_pipe = clone(models['Elastic Net']['ml_m'])

elnet_y_pipe.fit(df[X_cols], df['log_transport_co2'])
best_l1_y = elnet_y_pipe.steps[1][1].l1_ratio_

elnet_t_pipe.fit(df[X_cols], df['cp_active'])
best_l1_t = elnet_t_pipe.steps[1][1].l1_ratio_[0]

hyperparam_path = results_dir / 'optimal_hyperparameters.txt'
with open(hyperparam_path, 'w') as f:
    f.write("--- ELASTIC NET OPTIMAL HYPERPARAMETERS (FULL SAMPLE) ---\n")
    f.write("Grid Search Options: [0.1, 0.5, 0.9, 0.99]\n\n")
    f.write(f"Outcome Model (Log Transport CO2) Selected l1_ratio: {best_l1_y}\n")
    f.write(f"Treatment Model (CP Active) Selected l1_ratio: {best_l1_t}\n")

print(f"Success: Optimal hyperparameters saved to {hyperparam_path}")