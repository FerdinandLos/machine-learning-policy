import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Forces headless rendering, bypassing Tkinter threading bugs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

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
# Note: Ensure rmse_evaluation_raw.csv exists from your earlier nuisance validation steps
try:
    rmse_df = pd.read_csv(results_dir / 'rmse_evaluation_raw.csv')
    rmse_df = rmse_df[rmse_df['Model'] != 'Ensemble (Stacking)']
    has_rmse = True
except FileNotFoundError:
    print("  [!] RMSE file not found. Skipping Table 1.")
    has_rmse = False

# UPDATED: Load the new AIPW/Causal Forest results
dml_df = pd.read_csv(results_dir / 'dml_robustness_results_aipw.csv')

# ---------------------------------------------------------
# 2. Process and Export RMSE Evaluation Table
# ---------------------------------------------------------
if has_rmse:
    print("Formatting RMSE evaluation table...")
    
    # Define the exact models to keep for the thesis table
    models_to_keep = [
        'L1 (Lasso / Logit L1)', 
        'L2 (Ridge / Logit L2)', 
        'Elastic Net', 
        'Random Forest'
    ]
    
    # Filter the dataframe to only include the target models
    filtered_rmse_df = rmse_df[rmse_df['Model'].isin(models_to_keep)].copy()
    
    # Explicitly map the column names
    rename_dict = {
        "RMSE D (cp_active)": "RMSE D (CP)", 
        "RMSE D (lez_active)": "RMSE D (LEZ)", 
        "RMSE D (cp_x_lez)": "RMSE D (CP $\\times$ LEZ)"
    }
    filtered_rmse_df.rename(columns=rename_dict, inplace=True)

    # Set index and isolate numeric operations
    rmse_indexed = filtered_rmse_df.set_index('Model')
    
    # Calculate best models BEFORE altering the dataframe structure
    best_models = rmse_indexed.idxmin()
    
    # Round the numeric values safely
    rmse_formatted = rmse_indexed.round(4)
    
    # Transpose FIRST so the model names become columns and metrics become rows
    rmse_transposed = rmse_formatted.T
    
    # Add the 'Best Model' row directly to the transposed dataframe
    rmse_transposed['Best Model'] = best_models

    # CRITICAL FIX: Ensure the transposed index (the row labels) matches the clean LaTeX names
    rmse_transposed.index = rmse_transposed.index.map(lambda x: rename_dict.get(x, x))

    # Generate LaTeX code
    latex_table_rmse = rmse_transposed.to_latex(
        float_format="%.4f",
        caption="Cross-fitted RMSE for predicting Nuisance Parameters",
        label="tab:rmse_evaluation",
        column_format="l" + "c" * len(rmse_transposed.columns),
        escape=False
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
    'L1 (Lasso / Logit L1)': '(2) AIPW (Lasso)',
    'Causal Forest': '(3) Causal Forest'
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
        
    return f"${coef:.4f}{stars}$ ($p={pval:.3f}$)"

for prefix in ['Global_ATE', 'Global_ATT', 'GATT_Cluster_1', 'GATT_Cluster_0']:
    dml_core[f'{prefix}_fmt'] = dml_core.apply(
        lambda row: format_estimate(row[f'{prefix}_coef'], row[f'{prefix}_pval']), axis=1
    )

def get_row_vals(pol, prefix):
    row_vals = []
    for mod in target_models.keys():
        val = dml_core[(dml_core['Model'] == mod) & (dml_core['Policy'] == pol)][f'{prefix}_fmt'].values
        row_vals.append(val[0] if len(val) > 0 else "--")
    return row_vals

# Build clean manual LaTeX table lines to avoid pandas column-span conflicts
latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Double Machine Learning Estimates of Urban Climate Policies (Categorical Regime)}")
latex_lines.append(r"\label{tab:dml_master_results}")
latex_lines.append(r"\begin{tabular}{lccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Policy & (1) OLS Baseline & (2) AIPW (Lasso) & (3) Causal Forest \\")
latex_lines.append(r"\midrule")

# Panel A
latex_lines.append(r"\multicolumn{4}{l}{\textbf{Panel A: Global Average Treatment Effect (ATE)}} \\")
for pol, clean_name in policy_map.items():
    v = get_row_vals(pol, 'Global_ATE')
    latex_lines.append(f"\\hspace{{4mm}} {clean_name} & {v[0]} & {v[1]} & {v[2]} \\\\")

# Panel B
latex_lines.append(r"\addlinespace")
latex_lines.append(r"\multicolumn{4}{l}{\textbf{Panel B: Average Treatment Effect on the Treated (ATT)}} \\")
for pol, clean_name in policy_map.items():
    v = get_row_vals(pol, 'Global_ATT')
    latex_lines.append(f"\\hspace{{4mm}} {clean_name} & {v[0]} & {v[1]} & {v[2]} \\\\")

# Panel C
latex_lines.append(r"\addlinespace")
latex_lines.append(r"\multicolumn{4}{l}{\textbf{Panel C: Group ATT (Heterogeneity by City Type)}} \\")
latex_lines.append(r"\multicolumn{4}{l}{\textit{Cluster 1: Dense Metropolis}} \\")
for pol, clean_name in policy_map.items():
    v = get_row_vals(pol, 'GATT_Cluster_1')
    latex_lines.append(f"\\hspace{{4mm}} {clean_name} & {v[0]} & {v[1]} & {v[2]} \\\\")

latex_lines.append(r"\multicolumn{4}{l}{\textit{Cluster 0: Sprawling Cities}} \\")
for pol, clean_name in policy_map.items():
    v = get_row_vals(pol, 'GATT_Cluster_0')
    latex_lines.append(f"\\hspace{{4mm}} {clean_name} & {v[0]} & {v[1]} & {v[2]} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\raggedright \footnotesize \textit{Notes:} $p$-values in parentheses. $^{*} p < 0.10$, $^{**} p < 0.05$, $^{***} p < 0.01$. 'Failed' estimation bounds denote overlap violations.")
latex_lines.append(r"\end{table}")


with open(tables_dir / 'dml_master_results.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: dml_master_results.tex saved.")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# ---------------------------------------------------------
# 4. Figure 1: Coefficient Forest Plot (Global ATT)
# ---------------------------------------------------------
print("Generating Figure 1: Coefficient Forest Plot...")

def approx_se(coef, pval):
    if pd.isna(coef) or pd.isna(pval): 
        return 0
    p = max(pval, 1e-15) 
    z = np.abs(norm.ppf(p / 2))
    return abs(coef) / z

# UPDATED: Pulling directly from the Global_ATT columns
dml_core['Global_ATT_se'] = dml_core.apply(lambda row: approx_se(row['Global_ATT_coef'], row['Global_ATT_pval']), axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

y_labels = [policy_map[p] for p in core_policies]
y_ticks = np.arange(len(y_labels))

# UPDATED: Keys matching the new target_models dictionary
offsets = {'(1) OLS Baseline': -0.15, '(2) AIPW (Lasso)': 0, '(3) Causal Forest': 0.15}
colors = {'(1) OLS Baseline': 'gray', '(2) AIPW (Lasso)': '#d95f02', '(3) Causal Forest': '#1b9e77'}

for mod, clean_mod in target_models.items():
    subset = dml_core[dml_core['Model'] == mod].set_index('Policy').reindex(core_policies)
    y_pos = y_ticks + offsets[clean_mod]
    
    ax.errorbar(subset['Global_ATT_coef'], y_pos, xerr=subset['Global_ATT_se'] * 1.96, 
                fmt='o', label=clean_mod, color=colors[clean_mod], 
                capsize=4, elinewidth=2, markersize=8)

ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_xlabel("Average Treatment Effect on the Treated (ATT)\nLog Transport CO2")
ax.set_title("Figure 1: Policy Efficacy by Algorithm (95% CI)", pad=15, fontweight='bold')
ax.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
fig.savefig(figures_dir / 'forest_plot_att.pdf', format='pdf', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 5. Figure 2: Heterogeneity Bar Chart (GATT)
# ---------------------------------------------------------
print("Generating Figure 2: Heterogeneity Bar Chart...")

# UPDATED: Utilizing the champion AIPW model (Lasso) for the visualization
champion_df = dml_core[dml_core['Model'] == 'L1 (Lasso / Logit L1)'].set_index('Policy').reindex(core_policies)

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35
x = np.arange(len(core_policies))

# UPDATED: Pulling from the specific GATT columns
ax.bar(x - bar_width/2, champion_df['GATT_Cluster_0_coef'], bar_width, 
       label='Cluster 0 (Sprawling)', color='#a6cee3', edgecolor='black')
ax.bar(x + bar_width/2, champion_df['GATT_Cluster_1_coef'], bar_width, 
       label='Cluster 1 (Dense Metropolis)', color='#1f78b4', edgecolor='black')

ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(y_labels)
ax.set_ylabel("Group Average Treatment Effect on the Treated (GATT)")
ax.set_title("Figure 2: Urban Heterogeneity in Policy Impact (AIPW)", pad=15, fontweight='bold')
ax.legend()

plt.tight_layout()
fig.savefig(figures_dir / 'heterogeneity_bar_gatt.pdf', format='pdf')
plt.close()

# ---------------------------------------------------------
# 6. Figure 3: Common Support (Overlap) Density Plot
# ---------------------------------------------------------
print("Generating Figure 3: Common Support Density Plot...")

# Assumes propensity_scores_exact_aipw.csv is generated from the Lasso model
ps_df = pd.read_csv(results_dir / 'propensity_scores_exact_aipw.csv')

fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(ps_df.loc[ps_df['cp_active'] == 1, 'propensity_score'], fill=True, label="Treated (CP Active)", color="#d95f02", ax=ax, alpha=0.5)
sns.kdeplot(ps_df.loc[ps_df['cp_active'] == 0, 'propensity_score'], fill=True, label="Untreated (No CP)", color="#7570b3", ax=ax, alpha=0.5)

ax.set_xlim(0, 1)
ax.set_xlabel("Estimated Propensity Score $\\hat{P}(D=1|X)$")
ax.set_ylabel("Density")
ax.set_title("Figure 3: Common Support for Congestion Pricing (AIPW)", pad=15, fontweight='bold')
ax.legend()

plt.tight_layout()
fig.savefig(figures_dir / 'overlap_density.pdf', format='pdf')
plt.close()

# ---------------------------------------------------------
# Console Summary: Common Support Numerical Distribution
# ---------------------------------------------------------
ps_treated = ps_df.loc[ps_df['cp_active'] == 1, 'propensity_score']
ps_control = ps_df.loc[ps_df['cp_active'] == 0, 'propensity_score']

# 1. Summary Statistics Table
ps_summary = pd.DataFrame({
    'Treated (CP Active)': ps_treated.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]),
    'Untreated (No CP)': ps_control.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
}).round(4)

# 2. Compute Common Support Boundaries
overlap_min = max(ps_treated.min(), ps_control.min())
overlap_max = min(ps_treated.max(), ps_control.max())
n_treated_in_range = (ps_treated >= overlap_min) & (ps_treated <= overlap_max)
n_control_in_range = (ps_control >= overlap_min) & (ps_control <= overlap_max)

print("\n" + "="*70)
print("NUMERICAL PROPENSITY SCORE DISTRIBUTION (COMMON SUPPORT)")
print("="*70)
print(ps_summary.to_string())
print("-" * 70)
print(f"Common Support Range: [{overlap_min:.4f}, {overlap_max:.4f}]")
print(f"Treated Units in Support:   {n_treated_in_range.sum()} / {len(ps_treated)} ({n_treated_in_range.mean():.1%})")
print(f"Untreated Units in Support: {n_control_in_range.sum()} / {len(ps_control)} ({n_control_in_range.mean():.1%})")
print("="*70 + "\n")

# ---------------------------------------------------------
# Export Compact LaTeX Table: Propensity Score Distribution
# ---------------------------------------------------------
print("Exporting Compact Propensity Score Summary Table to LaTeX...")

# 1. Filter to only the essential rows and map clean names
rows_to_keep = ['count', 'mean', 'std', 'min', '50%', 'max']
ps_summary_compact = ps_summary.loc[rows_to_keep].copy()

index_mapping = {
    'count': 'Observations (N)',
    'mean': 'Mean',
    'std': 'Std. Dev.',
    'min': 'Minimum',
    '50%': 'Median',
    'max': 'Maximum'
}
ps_summary_compact = ps_summary_compact.rename(index=index_mapping)

# 2. Calculate percentages for the footnote
treated_pct = n_treated_in_range.mean() * 100
control_pct = n_control_in_range.mean() * 100

# 3. Build manual LaTeX lines
latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Propensity Score Distribution and Common Support}")
latex_lines.append(r"\label{tab:propensity_scores}")
latex_lines.append(r"\begin{tabular}{lcc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Statistic & Treated (CP Active) & Untreated (No CP) \\")
latex_lines.append(r"\midrule")

# Populate data rows
for index, row in ps_summary_compact.iterrows():
    if 'Observations' in index:
        val1 = f"{int(row['Treated (CP Active)'])}"
        val2 = f"{int(row['Untreated (No CP)'])}"
    else:
        val1 = f"{row['Treated (CP Active)']:.3f}"
        val2 = f"{row['Untreated (No CP)']:.3f}"
    
    latex_lines.append(f"{index} & {val1} & {val2} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")

# 4. Add the overlap metrics as a formatted footnote
latex_lines.append(r"") 
latex_lines.append(r"\vspace{1ex}")
latex_lines.append(r"{\raggedright \footnotesize \textit{Notes:} The common support region is strictly defined as $[%.4f, %.4f]$. " % (overlap_min, overlap_max))
latex_lines.append(r"Within this bounded interval, the analysis retains %d treated units (%.1f\%%) and %d untreated units (%.1f\%%).\par}" % (
    n_treated_in_range.sum(), treated_pct, n_control_in_range.sum(), control_pct
))
latex_lines.append(r"\end{table}")

# 5. Save file to disk
with open(tables_dir / 'propensity_score_summary.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: compact propensity_score_summary.tex saved.")

print(f"Success: All figures generated and saved as PDFs in {figures_dir}")