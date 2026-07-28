import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import mean_squared_error

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
tables_dir = Path('Writing/Tables')
tables_dir.mkdir(parents=True, exist_ok=True)

class ClippedProbaClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom wrapper to prevent tree-based models from outputting exact 0.0 or 1.0 probabilities.
    Ensures safe propensity score clipping for rare events.
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

csv_path = Path('Data/urban_emissions_panel_cleaned.csv')
df = pd.read_csv(csv_path)

# Ensure the exact same categorical architecture as the estimation pipeline
df['policy_regime'] = (df['cp_active'] * 1) + (df['lez_active'] * 2)

year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

exclude_from_X = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'policy_regime',
    # --- COMPOSITIONAL DUMMY TRAP FIX ---
    'industry_public', 'fleet_petrol_share'
]

numeric_df = df.select_dtypes(include=[np.number])
X_cols = [col for col in numeric_df.columns if col not in exclude_from_X]

X_arr = df[X_cols].to_numpy()
Y_arr = df['log_transport_co2'].to_numpy()
T_arr = df['policy_regime'].to_numpy()
city_groups = df['city_id'].to_numpy()

# ---------------------------------------------------------
# 2. Define the Machine Learning Learners (EconML Scope)
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
# 3. Calculate Cross-Fitted RMSE
# ---------------------------------------------------------
print("--- CALCULATING CROSS-FITTED RMSE FOR NUISANCE PARAMETERS ---")
results = []

# Enforce strictly grouped out-of-fold predictions to prevent panel data leakage
cv_panel = GroupKFold(n_splits=5)

for name, model_dict in models.items():
    print(f"Evaluating {name}...")
    reg = model_dict['ml_l']
    clf = model_dict['ml_m']
    
    # 1. Predict Y (Outcome Nuisance)
    preds_Y = cross_val_predict(reg, X_arr, Y_arr, cv=cv_panel, groups=city_groups)
    rmse_Y = np.sqrt(mean_squared_error(Y_arr, preds_Y))
    
    row_data = {
        'Model': name,
        'RMSE Y (Outcome)': rmse_Y,
    }
    
    # 2. Predict T (Multi-class Categorical Treatment Nuisance)
    # Scikit-learn outputs a probability matrix with columns for [Class 0, Class 1, Class 2, Class 3]
    preds_T_proba = cross_val_predict(clf, X_arr, T_arr, cv=cv_panel, groups=city_groups, method='predict_proba')
    
    # Extract binary truth arrays for calculating distinct regime RMSEs
    true_cp = (T_arr == 1).astype(int)
    true_lez = (T_arr == 2).astype(int)
    true_syn = (T_arr == 3).astype(int)

    row_data['RMSE D (cp_active)'] = np.sqrt(mean_squared_error(true_cp, preds_T_proba[:, 1]))
    row_data['RMSE D (lez_active)'] = np.sqrt(mean_squared_error(true_lez, preds_T_proba[:, 2]))
    row_data['RMSE D (cp_x_lez)'] = np.sqrt(mean_squared_error(true_syn, preds_T_proba[:, 3]))
        
    results.append(row_data)

# ---------------------------------------------------------
# 4. Export Raw Results to CSV
# ---------------------------------------------------------
master_df = pd.DataFrame(results).set_index('Model')

preview_df = master_df.copy()
best_models = preview_df.idxmin()
preview_df = preview_df.round(4)
preview_df.loc['Best Model'] = best_models

print("\n--- FINAL CROSS-FITTED RMSE EVALUATION ---")
print(preview_df.T.to_string())

results_dir = Path('Data/Results')
results_dir.mkdir(parents=True, exist_ok=True)
csv_export_path = results_dir / 'rmse_evaluation_raw.csv'
master_df.reset_index().to_csv(csv_export_path, index=False)

print(f"\nSuccess: Raw RMSE estimations safely exported to {csv_export_path}")