import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Forces headless rendering, bypassing Tkinter threading bugs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
results_dir = Path('Data/Results')
tables_dir = Path('Writing/Tables')
figures_dir = Path('Writing/Figures')

tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

print("Loading raw evaluation datasets...")
rmse_df = pd.read_csv(results_dir / 'rmse_evaluation_raw.csv')
rmse_df = rmse_df[rmse_df['Model'] != 'Ensemble (Stacking)']
dml_df = pd.read_csv(results_dir / 'dml_robustness_results_econml.csv')

# ---------------------------------------------------------
# 2. Process and Export RMSE Evaluation Table
# ---------------------------------------------------------
print("Formatting RMSE evaluation table...")
rmse_indexed = rmse_df.set_index('Model')
best_models = rmse_indexed.idxmin()
rmse_formatted = rmse_indexed.round(4)
rmse_formatted.loc['Best Model'] = best_models
rmse_transposed = rmse_formatted.T

latex_table_rmse = rmse_transposed.to_latex(
    float_format="%.4f",
    caption="Cross-fitted RMSE for predicting Nuisance Parameters",
    label="tab:rmse_evaluation",
    column_format="l" + "c" * len(rmse_formatted.index)
)

with open(tables_dir / 'rmse_evaluation_final.tex', 'w') as f: 
    f.write(latex_table_rmse)

print("Success: rmse_evaluation_final.tex saved.")

# ---------------------------------------------------------
# 3. Process and Export DML Master Table (Panels A, B, C)
# ---------------------------------------------------------
print("Formatting DML Master Causal Table...")

target_models = {
    'OLS - Basic': '(1) OLS Baseline',
    'Boosted Trees': '(2) Boosted Trees',
    'L1 (Lasso / Logit L1)': '(3) LASSO'
}

policy_map = {
    'cp_active': 'Congestion Pricing (CP)',
    'lez_active': 'Low Emission Zone (LEZ)',
    'cp_x_lez': 'Synergy (CP $\\times$ LEZ)'
}

dml_core = dml_df[dml_df['Model'].isin(target_models.keys())].copy()

def format_estimate(coef, pval):
    if pd.isna(coef) or pd.isna(pval):
        return "--" 
    
    stars = ""
    if pval < 0.01:
        stars = "^{***}"
    elif pval < 0.05:
        stars = "^{**}"
    elif pval < 0.10:
        stars = "^{*}"
        
    return f"{coef:.4f}{stars} (p={pval:.3f})"

# UPDATED: Replaced PLM/IRM strings with the correct Global_ATE and CATE strings
for prefix in ['Global_ATE', 'CATE_ATET', 'CATE_Cluster_1', 'CATE_Cluster_0']:
    dml_core[f'{prefix}_fmt'] = dml_core.apply(
        lambda row: format_estimate(row[f'{prefix}_coef'], row[f'{prefix}_pval']), axis=1
    )

master_table_rows = []

# --- PANEL A: Global ATE (Constant Partially Linear Model) ---
master_table_rows.append({'Policy': '\\textbf{Panel A: Constant Partially Linear Model (Global ATE)}', **{v: '' for v in target_models.values()}})
for pol, clean_name in policy_map.items():
    row = {'Policy': f"\\hspace{{4mm}} {clean_name}"}
    for mod, col in target_models.items():
        val = dml_core[(dml_core['Model'] == mod) & (dml_core['Policy'] == pol)]['Global_ATE_fmt'].values
        row[col] = val[0] if len(val) > 0 else "--"
    master_table_rows.append(row)

master_table_rows.append({'Policy': '', **{v: '' for v in target_models.values()}})

# --- PANEL B: CATE ATET ---
master_table_rows.append({'Policy': '\\textbf{Panel B: CATE Estimation (ATET for Adopters)}', **{v: '' for v in target_models.values()}})
for pol, clean_name in policy_map.items():
    row = {'Policy': f"\\hspace{{4mm}} {clean_name}"}
    for mod, col in target_models.items():
        val = dml_core[(dml_core['Model'] == mod) & (dml_core['Policy'] == pol)]['CATE_ATET_fmt'].values
        row[col] = val[0] if len(val) > 0 else "--"
    master_table_rows.append(row)

master_table_rows.append({'Policy': '', **{v: '' for v in target_models.values()}})

# --- PANEL C: Intra-Cluster CATE ---
master_table_rows.append({'Policy': '\\textbf{Panel C: Intra-Cluster CATE (Heterogeneity by City Type)}', **{v: '' for v in target_models.values()}})
master_table_rows.append({'Policy': '\\textit{Cluster 1: Dense Metropolis}', **{v: '' for v in target_models.values()}})
for pol, clean_name in policy_map.items():
    row = {'Policy': f"\\hspace{{4mm}} {clean_name}"}
    for mod, col in target_models.items():
        val = dml_core[(dml_core['Model'] == mod) & (dml_core['Policy'] == pol)]['CATE_Cluster_1_fmt'].values
        row[col] = val[0] if len(val) > 0 else "--"
    master_table_rows.append(row)

master_table_rows.append({'Policy': '\\textit{Cluster 0: Sprawling Cities}', **{v: '' for v in target_models.values()}})
for pol, clean_name in policy_map.items():
    row = {'Policy': f"\\hspace{{4mm}} {clean_name}"}
    for mod, col in target_models.items():
        val = dml_core[(dml_core['Model'] == mod) & (dml_core['Policy'] == pol)]['CATE_Cluster_0_fmt'].values
        row[col] = val[0] if len(val) > 0 else "--"
    master_table_rows.append(row)

final_dml_df = pd.DataFrame(master_table_rows)

latex_table_dml = final_dml_df.to_latex(
    index=False,
    escape=False,
    column_format="l" + "c" * len(target_models),
    caption="Double Machine Learning Estimates of Urban Climate Policies",
    label="tab:dml_master_results"
)

with open(tables_dir / 'dml_master_results.tex', 'w') as f:
    f.write(latex_table_dml)
    f.write("\\raggedright \\footnotesize \\textit{Notes:} $p$-values in parentheses. $^{*}$ $p < 0.10$, $^{**}$ $p < 0.05$, $^{***}$ $p < 0.01$. 'Failed' estimation bounds denote overlap violations.\n")

print("Success: dml_master_results.tex saved.")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# ---------------------------------------------------------
# 4. Figure 1: Coefficient Forest Plot (CATE ATET)
# ---------------------------------------------------------
print("Generating Figure 1: Coefficient Forest Plot...")

def approx_se(coef, pval):
    if pd.isna(coef) or pd.isna(pval): 
        return 0
    p = max(pval, 1e-15) 
    z = np.abs(norm.ppf(p / 2))
    return abs(coef) / z

# UPDATED: Pulling from CATE_ATET
dml_core['CATE_ATET_se'] = dml_core.apply(lambda row: approx_se(row['CATE_ATET_coef'], row['CATE_ATET_pval']), axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

y_labels = [policy_map[p] for p in core_policies]
y_ticks = np.arange(len(y_labels))

offsets = {'(1) OLS Baseline': -0.15, '(2) Boosted Trees': 0, '(3) LASSO': 0.15}
colors = {'(1) OLS Baseline': 'gray', '(2) Boosted Trees': '#d95f02', '(3) LASSO': '#1b9e77'}

for mod, clean_mod in target_models.items():
    subset = dml_core[dml_core['Model'] == mod].set_index('Policy').reindex(core_policies)
    y_pos = y_ticks + offsets[clean_mod]
    
    ax.errorbar(subset['CATE_ATET_coef'], y_pos, xerr=subset['CATE_ATET_se'] * 1.96, 
                fmt='o', label=clean_mod, color=colors[clean_mod], 
                capsize=4, elinewidth=2, markersize=8)

ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
# UPDATED: Y-axis label to match new terminology
ax.set_xlabel("Conditional Average Treatment Effect for Adopters (CATE-ATET)\nLog Transport CO2")
ax.set_title("Figure 1: Policy Efficacy by Algorithm (95% CI)", pad=15, fontweight='bold')
ax.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
fig.savefig(figures_dir / 'forest_plot_atet.pdf', format='pdf', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 5. Figure 2: Heterogeneity Bar Chart (CATE)
# ---------------------------------------------------------
print("Generating Figure 2: Heterogeneity Bar Chart...")

# BUG FIX: The model key is 'L1 (Lasso / Logit L1)', not 'LASSO'
champion_df = dml_core[dml_core['Model'] == 'L1 (Lasso / Logit L1)'].set_index('Policy').reindex(core_policies)

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35
x = np.arange(len(core_policies))

# UPDATED: Pulling from CATE_Cluster
ax.bar(x - bar_width/2, champion_df['CATE_Cluster_0_coef'], bar_width, 
       label='Cluster 0 (Sprawling)', color='#a6cee3', edgecolor='black')
ax.bar(x + bar_width/2, champion_df['CATE_Cluster_1_coef'], bar_width, 
       label='Cluster 1 (Dense Metropolis)', color='#1f78b4', edgecolor='black')

ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(y_labels)
# UPDATED: Y-axis label
ax.set_ylabel("Conditional Average Treatment Effect (CATE)")
ax.set_title("Figure 2: Urban Heterogeneity in Policy Impact (LASSO)", pad=15, fontweight='bold')
ax.legend()

plt.tight_layout()
fig.savefig(figures_dir / 'heterogeneity_bar_gatet.pdf', format='pdf')
plt.close()

# ---------------------------------------------------------
# 6. Figure 3: Common Support (Overlap) Density Plot
# ---------------------------------------------------------
print("Generating Figure 3: Common Support Density Plot...")

ps_df = pd.read_csv(results_dir / 'propensity_scores_exact.csv')

fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(ps_df.loc[ps_df['cp_active'] == 1, 'propensity_score'], fill=True, label="Treated (CP Active)", color="#d95f02", ax=ax, alpha=0.5)
sns.kdeplot(ps_df.loc[ps_df['cp_active'] == 0, 'propensity_score'], fill=True, label="Untreated (No CP)", color="#7570b3", ax=ax, alpha=0.5)

ax.set_xlim(0, 1)
ax.set_xlabel("Estimated Propensity Score $\\hat{P}(D=1|X)$")
ax.set_ylabel("Density")
ax.set_title("Figure 3: Common Support for Congestion Pricing (LASSO)", pad=15, fontweight='bold')
ax.legend()

plt.tight_layout()
fig.savefig(figures_dir / 'overlap_density.pdf', format='pdf')
plt.close()

print(f"Success: All figures generated and saved as PDFs in {figures_dir}")