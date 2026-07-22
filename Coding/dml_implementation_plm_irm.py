import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegressionCV
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

# Create the synergy and heterogeneity interaction terms
df['cp_x_lez'] = df['cp_active'] * df['lez_active']
df['cp_x_type1'] = df['cp_active'] * df['cluster_id']
df['lez_x_type1'] = df['lez_active'] * df['cluster_id']

# Create year dummies
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define X (Covariates)
exclude_from_X = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'cp_x_type1', 'lez_x_type1', 'cluster_id'
]
numeric_df = df.select_dtypes(include=[np.number])
X_cols = [col for col in numeric_df.columns if col not in exclude_from_X]

X = df[X_cols]
Y = df['log_transport_co2']

# Define the specific Treatment Vectors for Model 1 and Model 2
D_cols_m1 = ['cp_active', 'lez_active', 'cp_x_lez']
D_cols_m2 = ['cp_active', 'lez_active', 'cp_x_type1', 'lez_x_type1']

# All unique D columns we need to predict across both models
all_D_cols = list(set(D_cols_m1 + D_cols_m2))

# ---------------------------------------------------------
# 2. Define the Machine Learning Learners
# ---------------------------------------------------------
models = {
    'OLS - Basic': {
        'regressor': make_pipeline(StandardScaler(), LinearRegression()),
        'classifier': make_pipeline(StandardScaler(), LinearRegression()) 
    },
    'Lasso / Logistic CV': {
        'regressor': make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=1)),
        'classifier': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', random_state=42, max_iter=10000, n_jobs=1))
    },
    'Random Forest': {
        'regressor': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1),
        'classifier': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
    },
    'Boosted Trees (Depth 5)': {
        'regressor': HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5),
        'classifier': HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5)
    }
}

# ---------------------------------------------------------
# 3. Calculate Cross-Fitted RMSE
# ---------------------------------------------------------
print("--- CALCULATING CROSS-FITTED RMSE FOR ALL MODELS ---")
results = []

for name, model_dict in models.items():
    print(f"Evaluating {name}...")
    reg = model_dict['regressor']
    clf = model_dict['classifier']
    
    # 1. RMSE Y (PLM): Predict Y using only X (Shared baseline for all parameters)
    preds_Y_plm = cross_val_predict(reg, X, Y, cv=5, n_jobs=5)
    rmse_Y_plm = np.sqrt(mean_squared_error(Y, preds_Y_plm))
    
    row_data = {
        'Model': name,
        'RMSE Y (PLM)': rmse_Y_plm,
    }
    
    # 2. Iterate through every single policy variable to evaluate its specific nuisance functions
    for d_col in all_D_cols:
        target_D = df[d_col]
        
        # A. Predict the Treatment (Propensity Score for both ATE and ATET)
        if name == 'OLS - Basic':
            preds_D = cross_val_predict(clf, X, target_D, cv=5, n_jobs=5)
        else:
            preds_D = cross_val_predict(clf, X, target_D, cv=5, method='predict_proba', n_jobs=5)[:, 1]
        row_data[f'RMSE D ({d_col})'] = np.sqrt(mean_squared_error(target_D, preds_D))
        
        # B. Predict the Outcome for IRM ATE (Trained/Tested on full sample using X and D)
        X_and_D_single = df[X_cols + [d_col]]
        preds_Y_ate = cross_val_predict(reg, X_and_D_single, Y, cv=5, n_jobs=5)
        row_data[f'RMSE Y IRM ATE ({d_col})'] = np.sqrt(mean_squared_error(Y, preds_Y_ate))
        
        # C. Predict the Outcome for IRM ATET (Trained/Tested strictly on D=0 subset using X)
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

# Add 'Best' ensemble row
master_df.loc['Best'] = master_df.min()
master_df = master_df.round(4)

# --------------------------------
# Split into Model 1 Output Matrix
# --------------------------------
cols_m1 = ['RMSE Y (PLM)']
for d_col in D_cols_m1:
    cols_m1.extend([f'RMSE D ({d_col})', f'RMSE Y IRM ATE ({d_col})', f'RMSE Y IRM ATET ({d_col})'])

df_m1 = master_df[cols_m1]

# --------------------------------
# Split into Model 2 Output Matrix
# --------------------------------
cols_m2 = ['RMSE Y (PLM)']
for d_col in D_cols_m2:
    cols_m2.extend([f'RMSE D ({d_col})', f'RMSE Y IRM ATE ({d_col})', f'RMSE Y IRM ATET ({d_col})'])

df_m2 = master_df[cols_m2]

print("\n--- CROSS-FITTED RMSE: MODEL 1 (AVERAGE EFFECTS & SYNERGY) ---")
print(df_m1.T.to_string())

print("\n--- CROSS-FITTED RMSE: MODEL 2 (HETEROGENEITY) ---")
print(df_m2.T.to_string())

# Export Model 1 to LaTeX (Transposed so Algorithms are columns, Metrics are rows)
latex_m1 = df_m1.T.to_latex(
    float_format="%.4f",
    caption="Model 1: Cross-fitted RMSE for predicting ATE and ATET Nuisance Parameters",
    label="tab:rmse_m1",
    column_format="l" + "c" * len(df_m1.index)
)
with open(tables_dir / 'rmse_evaluation_m1.tex', 'w') as f: f.write(latex_m1)

# Export Model 2 to LaTeX (Transposed)
latex_m2 = df_m2.T.to_latex(
    float_format="%.4f",
    caption="Model 2: Cross-fitted RMSE for predicting ATE and ATET Nuisance Parameters",
    label="tab:rmse_m2",
    column_format="l" + "c" * len(df_m2.index)
)
with open(tables_dir / 'rmse_evaluation_m2.tex', 'w') as f: f.write(latex_m2)

print("\nSuccess: Both RMSE evaluation matrices transposed and securely exported to Writing/Tables/")