import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from econml.dr import LinearDRLearner

warnings.filterwarnings('ignore')

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
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime', 'industry_public', 'fleet_petrol_share'
]
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]

cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='Cluster', dtype=int)
cluster_dummies_array = cluster_dummies.to_numpy()

# ---------------------------------------------------------
# 2. Sensitivity Loop Setup
# ---------------------------------------------------------
print("--- INITIATING OVERLAP & POSITIVITY SENSITIVITY ---")

cv_panel = GroupKFold(n_splits=5)
regime_mapping = {'cp_active': 1, 'lez_active': 2, 'cp_x_lez': 3}

# The trimming thresholds to test (from highly permissive to highly aggressive)
# 0.001 means keeping almost everything; 0.10 means dropping a massive amount of data
trim_thresholds = [0.001, 0.01, 0.02, 0.05, 0.10]

# Champion Model: Lasso (L1)
model_name = 'L1 (Lasso / Logit L1)'
ml_l = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1))
ml_m = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l1', solver='saga', scoring='neg_log_loss', random_state=42, max_iter=10000, n_jobs=-1))

W_matrix = df[base_W_cols].to_numpy()
T_arr = df['policy_regime'].to_numpy()
Y_arr = df['log_transport_co2'].to_numpy()
city_groups = df['city_id'].to_numpy()

sensitivity_results = []

# ---------------------------------------------------------
# 3. Execute Trimming Battery
# ---------------------------------------------------------
for trim in trim_thresholds:
    print(f"\n>> Testing strictness threshold: min_propensity = {trim}")
    
    try:
        estimator = LinearDRLearner(
            model_regression=clone(ml_l),
            model_propensity=clone(ml_m),
            min_propensity=trim,  
            fit_cate_intercept=False, # --- BUG 1 FIX: Prevent rank-deficiency ---
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
            row_data = {'Model': model_name, 'Trim_Threshold': trim, 'Policy': policy_name}
            
            # Global ATE
            row_data['Global_ATE_coef'] = np.atleast_1d(estimator.ate(X=cluster_dummies_array, T0=0, T1=t_val))[0]
            row_data['Global_ATE_pval'] = np.atleast_1d(estimator.ate_inference(X=cluster_dummies_array, T0=0, T1=t_val).pvalue())[0]

            # Global ATT
            treated_mask = (df['policy_regime'].to_numpy() == t_val)
            X_treated = cluster_dummies_array[treated_mask]
            
            row_data['Global_ATT_coef'] = np.atleast_1d(estimator.ate(X=X_treated, T0=0, T1=t_val))[0]
            row_data['Global_ATT_pval'] = np.atleast_1d(estimator.ate_inference(X=X_treated, T0=0, T1=t_val).pvalue())[0]
            
            sensitivity_results.append(row_data)

    except Exception as e:
        print(f"   [!] Estimation failed for threshold {trim}: {e}")

# ---------------------------------------------------------
# 4. Export Results for Plotting
# ---------------------------------------------------------
sensitivity_df = pd.DataFrame(sensitivity_results)
export_path = results_dir / 'sensitivity_overlap_trimming.csv'
sensitivity_df.to_csv(export_path, index=False)

print(f"\nSuccess: Overlap sensitivity data safely exported to {export_path}")