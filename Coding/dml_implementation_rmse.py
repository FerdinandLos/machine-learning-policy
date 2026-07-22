import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
tables_dir = Path('Writing/Tables')
tables_dir.mkdir(parents=True, exist_ok=True)

csv_path = Path('Data/urban_emissions_panel_cleaned.csv')
df = pd.read_csv(csv_path)

# Create the synergy interaction term (We drop the type interactions)
df['cp_x_lez'] = df['cp_active'] * df['lez_active']

# Create year dummies
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define X (Covariates)
exclude_from_X = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez'
]
numeric_df = df.select_dtypes(include=[np.number])
X_cols = [col for col in numeric_df.columns if col not in exclude_from_X]

X = df[X_cols]
Y = df['log_transport_co2']

# Define the Core Policy Variables
core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

# ---------------------------------------------------------
# 2. Define the Machine Learning Learners
# ---------------------------------------------------------
# Define the base estimators for the Ensemble first
base_regressors = [
    ('elnet', make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], random_state=42, n_jobs=1))),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)),
    ('boost', HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5))
]

base_classifiers = [
    ('elnet', make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='elasticnet', solver='saga', l1_ratios=[0.1, 0.5, 0.9], random_state=42, max_iter=10000, n_jobs=1))),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)),
    ('boost', HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5))
]

# The complete grid of models to evaluate
models = {
    'OLS - Basic': {
        'regressor': make_pipeline(StandardScaler(), LinearRegression()),
        'classifier': make_pipeline(StandardScaler(), LinearRegression()) 
    },
    'L1 (Lasso / Logistic L1)': {
        'regressor': make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=1)),
        'classifier': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', random_state=42, max_iter=10000, n_jobs=1))
    },
    'L2 (Ridge / Logistic L2)': {
        'regressor': make_pipeline(StandardScaler(), RidgeCV(cv=5)),
        # lbfgs is the standard, fast solver for L2 penalties
        'classifier': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs', random_state=42, max_iter=10000, n_jobs=1))
    },
    'Elastic Net': {
        'regressor': make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], random_state=42, max_iter=10000, n_jobs=1)),
        # saga solver is required to handle elasticnet penalties in scikit-learn
        'classifier': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='elasticnet', solver='saga', l1_ratios=[0.1, 0.5, 0.9], random_state=42, max_iter=10000, n_jobs=1))
    },
    'Single Tree (Depth 5)': {
        'regressor': DecisionTreeRegressor(max_depth=5, random_state=42),
        'classifier': DecisionTreeClassifier(max_depth=5, random_state=42)
    },
    'Random Forest': {
        'regressor': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1),
        'classifier': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
    },
    'Boosted Trees': {
        'regressor': HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5),
        'classifier': HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5)
    },
    'Ensemble (Stacking)': {
        # Combines the predictions of Elastic Net, Random Forest, and Boosted Trees using a Ridge (L2) meta-learner to prevent overfitting
        'regressor': StackingRegressor(estimators=base_regressors, final_estimator=make_pipeline(StandardScaler(), RidgeCV(cv=5))),
        'classifier': StackingClassifier(estimators=base_classifiers, final_estimator=make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs')))
    }
}

# ---------------------------------------------------------
# 3. Calculate Cross-Fitted RMSE
# ---------------------------------------------------------
print("--- CALCULATING CROSS-FITTED RMSE FOR CORE POLICIES ---")
results = []

for name, model_dict in models.items():
    print(f"Evaluating {name}...")
    reg = model_dict['regressor']
    clf = model_dict['classifier']
    
    # 1. RMSE Y (PLM): Predict Y using only X
    preds_Y_plm = cross_val_predict(reg, X, Y, cv=5, n_jobs=5)
    rmse_Y_plm = np.sqrt(mean_squared_error(Y, preds_Y_plm))
    
    row_data = {
        'Model': name,
        'RMSE Y (PLM)': rmse_Y_plm,
    }
    
    # 2. Iterate through core policy variables
    for d_col in core_policies:
        target_D = df[d_col]
        
        # A. Predict the Treatment
        if name == 'OLS - Basic':
            preds_D = cross_val_predict(clf, X, target_D, cv=5, n_jobs=5)
        else:
            preds_D = cross_val_predict(clf, X, target_D, cv=5, method='predict_proba', n_jobs=5)[:, 1]
        row_data[f'RMSE D ({d_col})'] = np.sqrt(mean_squared_error(target_D, preds_D))
        
        # B. Predict Outcome for IRM ATE
        X_and_D_single = df[X_cols + [d_col]]
        preds_Y_ate = cross_val_predict(reg, X_and_D_single, Y, cv=5, n_jobs=5)
        row_data[f'RMSE Y IRM ATE ({d_col})'] = np.sqrt(mean_squared_error(Y, preds_Y_ate))
        
        # C. Predict Outcome for IRM ATET
        mask_untreated = target_D == 0
        X_untreated = X[mask_untreated]
        Y_untreated = Y[mask_untreated]
        preds_Y_atet = cross_val_predict(reg, X_untreated, Y_untreated, cv=5, n_jobs=5)
        row_data[f'RMSE Y IRM ATET ({d_col})'] = np.sqrt(mean_squared_error(Y_untreated, preds_Y_atet))
        
    results.append(row_data)

# ---------------------------------------------------------
# 4. Generate Output Matrices and LaTeX Tables
# ---------------------------------------------------------
master_df = pd.DataFrame(results).set_index('Model')

# Find the name of the model with the lowest RMSE BEFORE rounding
best_models = master_df.idxmin()

# Round numeric values to 4 decimal places
master_df = master_df.round(4)

# Append the text names as the final row (which becomes the final column when transposed)
master_df.loc['Best Model'] = best_models

print("\n--- FINAL CROSS-FITTED RMSE EVALUATION ---")
print(master_df.T.to_string())

# Export to LaTeX (Transposed)
latex_table = master_df.T.to_latex(
    float_format="%.4f",
    caption="Cross-fitted RMSE for predicting ATE and ATET Nuisance Parameters",
    label="tab:rmse_evaluation",
    column_format="l" + "c" * len(master_df.index)
)

with open(tables_dir / 'rmse_evaluation_final.tex', 'w') as f: 
    f.write(latex_table)

print("\nSuccess: Final RMSE evaluation matrix saved to Writing/Tables/rmse_evaluation_final.tex")