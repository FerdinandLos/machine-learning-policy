import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from doubleml import DoubleMLData, DoubleMLPLR

# ---------------------------------------------------------
# 0. System Setup & Data Loading
# ---------------------------------------------------------
# Define and create output directories to prevent FileNotFoundError
tables_dir = Path('Writing/Tables')
figures_dir = Path('Writing/Figures')
tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Load the newly cleaned urban emissions panel dataset
csv_path = Path('Data/urban_emissions_panel_cleaned.csv')
df = pd.read_csv(csv_path)

print("Dataset successfully loaded. Shape:", df.shape)

# ---------------------------------------------------------
# 1. Data Preparation & Feature Engineering
# ---------------------------------------------------------

# Create the synergy interaction term for Model 1
df['cp_x_lez'] = df['cp_active'] * df['lez_active']

# Create the heterogeneity interaction terms for Model 2
df['cp_x_type1'] = df['cp_active'] * df['cluster_id']
df['lez_x_type1'] = df['lez_active'] * df['cluster_id']

# Create year dummies to account for macro time trends
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)

# Define the confounder matrix (X)
# Note: log_transport_co2_initial is intentionally NOT excluded, as it serves as a baseline control
exclude_from_X = [
    'city_id', 'year', 'log_transport_co2', 'log_total_co2', 
    'cp_active', 'lez_active', 'cp_impl_year', 'lez_impl_year', 
    'cp_announce_year', 'lez_announce_year', 'country_id',
    'cp_x_lez', 'cp_x_type1', 'lez_x_type1', 'cluster_id'
]

# Ensure we only pick up numeric columns for the ML algorithms
numeric_df = df.select_dtypes(include=[np.number])
X_cols = [col for col in numeric_df.columns if col not in exclude_from_X]

# ---------------------------------------------------------
# 2. Setup Baseline Machine Learning Learners (Random Forests)
# ---------------------------------------------------------
ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

# ---------------------------------------------------------
# 3. BASELINE ESTIMATION
# ---------------------------------------------------------
print("\n--- ESTIMATING BASELINE (Random Forests) ---")
print("Model 1: Main Effects and Synergy...")
D_cols_m1 = ['cp_active', 'lez_active', 'cp_x_lez']
dml_data_m1 = DoubleMLData(df, y_col='log_transport_co2', d_cols=D_cols_m1, x_cols=X_cols, cluster_cols='city_id')
dml_plr_m1 = DoubleMLPLR(dml_data_m1, ml_l=clone(ml_l), ml_m=clone(ml_m), n_folds=5)
dml_plr_m1.fit()

print("Model 2: Heterogeneity by City Type...")
D_cols_m2 = ['cp_active', 'lez_active', 'cp_x_type1', 'lez_x_type1']
dml_data_m2 = DoubleMLData(df, y_col='log_transport_co2', d_cols=D_cols_m2, x_cols=X_cols, cluster_cols='city_id')
dml_plr_m2 = DoubleMLPLR(dml_data_m2, ml_l=clone(ml_l), ml_m=clone(ml_m), n_folds=5)
dml_plr_m2.fit()

# ---------------------------------------------------------
# 4. BASELINE OUTPUTS (Table and Plot)
# ---------------------------------------------------------
# FIXED: Replaced is_sensitivity boolean with dynamic learner_name
def generate_latex_table(model1, model2, original_df, filename, table_title, learner_name):
    """Reusable function to generate standard and sensitivity LaTeX tables."""
    df_m1, df_m2 = model1.summary, model2.summary
    var_order = ['cp_active', 'lez_active', 'cp_x_lez', 'cp_x_type1', 'lez_x_type1']
    var_mapping = {
        'cp_active': 'Congestion Pricing (CP)',
        'lez_active': 'Low-Emission Zone (LEZ)',
        'cp_x_lez': 'CP $\\times$ LEZ (Synergy)',
        'cp_x_type1': 'CP $\\times$ Dense Metropolis',
        'lez_x_type1': 'LEZ $\\times$ Dense Metropolis'
    }
    
    def format_cell(model_summary, variable):
        if variable not in model_summary.index: return "", ""
        coef, se, pval = model_summary.loc[variable, ['coef', 'std err', 'P>|t|']]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        return f"{coef:.2f}{stars}", f"({se:.2f})"

    table_body = ""
    for var in var_order:
        c1, s1 = format_cell(df_m1, var)
        c2, s2 = format_cell(df_m2, var)
        table_body += f"{var_mapping[var]} & {c1} & {c2} \\\\\n & {s1} & {s2} \\\\[0.4em]\n"

    n_obs = f"{original_df.shape[0]:,}"
    
    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{table_title}}}
\\label{{tab:{filename.split('.')[0]}}}
\\begin{{tabular}}{{lcc}}
\\hline\\hline
 & \\textbf{{Model 1: Average Effects}} & \\textbf{{Model 2: Heterogeneity}} \\\\
\\textbf{{Variable}} & \\textit{{(Main \\& Synergy)}} & \\textit{{(by City Type)}} \\\\
\\hline
{table_body}\\hline
ML Learner & {learner_name} & {learner_name} \\\\
Cross-Fitting Folds & {model1.n_folds} & {model1.n_folds} \\\\
Observations & {n_obs} & {n_obs} \\\\
\\hline\\hline
\\multicolumn{{3}}{{p{{12.5cm}}}}{{\\footnotesize \\textit{{Notes:}} Standard errors clustered at the city level in parentheses. * $p<0.10$, ** $p<0.05$, *** $p<0.01$. The dependent variable is log transport $CO_2$.}}
\\end{{tabular}}
\\end{{table}}
"""
    file_path = tables_dir / filename
    with open(file_path, 'w') as f: f.write(latex_code)
    print(f"Success: Saved {filename}")

def plot_forest(m1_res, m2_res, filename, title, marker_style='o'):
    labels = ['CP (Average)', 'LEZ (Average)', 'CP x LEZ (Synergy)', 
              'CP (Sprawling Hub Baseline)', 'CP x Type 1 (Dense Metropolis Diff)']
    
    # Extract coefficients and bounds
    coefs = [m1_res.loc['cp_active', 'coef'], m1_res.loc['lez_active', 'coef'], m1_res.loc['cp_x_lez', 'coef'],
             m2_res.loc['cp_active', 'coef'], m2_res.loc['cp_x_type1', 'coef']]
    errs = [coefs[0] - m1_res.loc['cp_active', '2.5 %'], coefs[1] - m1_res.loc['lez_active', '2.5 %'], 
            coefs[2] - m1_res.loc['cp_x_lez', '2.5 %'], coefs[3] - m2_res.loc['cp_active', '2.5 %'], 
            coefs[4] - m2_res.loc['cp_x_type1', '2.5 %']]

    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(coefs, range(len(labels))[::-1], xerr=errs, fmt=marker_style, color='black', capsize=5, capthick=1.5, elinewidth=1.5)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Estimated Effect on Log Transport $CO_2$ Emissions')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, format='pdf', bbox_inches='tight')
    plt.close()

# Generate Outputs
generate_latex_table(dml_plr_m1, dml_plr_m2, df, 'baseline_results_table.tex', 
                     'Double ML Causal Estimates on Log Transport $CO_2$ Emissions', 'Random Forest')
plot_forest(dml_plr_m1.summary, dml_plr_m2.summary, 'forest_plot_baseline.pdf', 'Baseline Treatment Effects (Random Forest)')


# ---------------------------------------------------------
# 5. SENSITIVITY ESTIMATION 1 (Penalized Linear Models)
# ---------------------------------------------------------
lasso_learner = make_pipeline(
    StandardScaler(),
    LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=1) 
)

logistic_learner = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', 
                         random_state=42, max_iter=10000, n_jobs=1)
)

print("\n--- ESTIMATING SENSITIVITY 1 (Lasso / Logistic) ---")
print("Model 1: Main Effects and Synergy...")
dml_plr_m1_sens = DoubleMLPLR(dml_data_m1, ml_l=lasso_learner, ml_m=logistic_learner, n_folds=5)
dml_plr_m1_sens.fit(n_jobs_cv=5)

print("Model 2: Heterogeneity by City Type...")
dml_plr_m2_sens = DoubleMLPLR(dml_data_m2, ml_l=lasso_learner, ml_m=logistic_learner, n_folds=5)
dml_plr_m2_sens.fit(n_jobs_cv=5)

generate_latex_table(dml_plr_m1_sens, dml_plr_m2_sens, df, 'sensitivity_lasso_table.tex', 
                     'Sensitivity Analysis: Penalized Linear Models', 'Lasso / Logistic CV')


# ---------------------------------------------------------
# 6. SENSITIVITY ESTIMATION 2 (Gradient Boosting)
# ---------------------------------------------------------
# Added to natively handle the K-Means collinearity trap that Lasso fails on
boost_l = HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5)
boost_m = HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=5)

print("\n--- ESTIMATING SENSITIVITY 2 (Gradient Boosting) ---")
print("Model 1: Main Effects and Synergy...")
dml_plr_m1_boost = DoubleMLPLR(dml_data_m1, ml_l=boost_l, ml_m=boost_m, n_folds=5)
dml_plr_m1_boost.fit(n_jobs_cv=5)

print("Model 2: Heterogeneity by City Type...")
dml_plr_m2_boost = DoubleMLPLR(dml_data_m2, ml_l=boost_l, ml_m=boost_m, n_folds=5)
dml_plr_m2_boost.fit(n_jobs_cv=5)

generate_latex_table(dml_plr_m1_boost, dml_plr_m2_boost, df, 'sensitivity_boost_table.tex', 
                     'Sensitivity Analysis: Gradient Boosting', 'HistGradientBoosting')

plot_forest(dml_plr_m1_boost.summary, dml_plr_m2_boost.summary, 'forest_plot_sensitivity.pdf', 
            'Sensitivity Estimates (Gradient Boosting)', marker_style='s')

print("\nAll estimations complete. Tables and figures successfully saved to your Writing folder.")

# ---------------------------------------------------------
# 7. DIAGNOSTIC: L1-PENALTY RESIDUAL COLLAPSE
# ---------------------------------------------------------
print("\n--- DIAGNOSTIC: L1-PENALTY ON HETEROGENEOUS TREATMENT ---")

# Fit exact logistic pipeline directly on the treatment variable
diagnostic_model = logistic_learner.fit(df[X_cols], df['cp_x_type1'])
fitted_logistic = diagnostic_model.named_steps['logisticregressioncv']
coefs = fitted_logistic.coef_[0]

coef_df = pd.DataFrame({'Covariate': X_cols, 'L1_Coefficient': coefs})

# Filter out the variables dropped to exactly 0.00
kept_vars = coef_df[coef_df['L1_Coefficient'] != 0.0].sort_values(by='L1_Coefficient', key=abs, ascending=False)
dropped_count = (coef_df['L1_Coefficient'] == 0.0).sum()

print(f"\nTotal variables dropped to exactly zero by L1 penalty: {dropped_count} out of {len(X_cols)}")
print("\nVariables KEPT by the model to predict 'cp_x_type1':")
print(kept_vars.to_string(index=False))

# Plot the Residual Collapse
predicted_probabilities = diagnostic_model.predict_proba(df[X_cols])[:, 1]

plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(predicted_probabilities, bins=40, color='darkred', edgecolor='black', alpha=0.7)
ax.set_title('Diagnostic: Predicted Probability of cp_x_type1 = 1')
ax.set_xlabel('Predicted Probability (Propensity Score)')
ax.set_ylabel('Number of City-Year Observations')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.axvline(0.05, color='black', linestyle=':', alpha=0.8)
ax.axvline(0.95, color='black', linestyle=':', alpha=0.8)
ax.text(0.05, ax.get_ylim()[1]*0.85, ' Near 0 \n (No Variation)', fontsize=9, ha='right')
ax.text(0.95, ax.get_ylim()[1]*0.85, ' Near 1 \n (No Variation)', fontsize=9, ha='left')

plt.tight_layout()
plt.savefig(figures_dir / 'diagnostic_overlap_violation.pdf', format='pdf', bbox_inches='tight')
plt.close()