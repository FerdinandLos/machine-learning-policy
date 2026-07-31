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
# 3. Process and Export DML Master Table (Panels A, B, C)
# ---------------------------------------------------------
print("Formatting DML Master Causal Table...")

target_models = {
    'L1 (Lasso / Logit L1)': '(1) AIPW',
    'Causal Forest': '(2) Causal Forest'
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

for prefix in ['Global_ATT', 'GATT_Cluster_1', 'GATT_Cluster_0']:
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
latex_lines.append(r"\caption{Estimates of Policy Effect on $Y=$ \textit{Transport $CO_2$ emissions}}")
latex_lines.append(r"\label{tab:dml_master_results}")
latex_lines.append(r"\begin{tabular}{lcc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Policy & (1) AIPW & (2) Causal Forest \\")
latex_lines.append(r"\midrule")

# Panel A (Keeps Synergy)
latex_lines.append(r"\addlinespace")
latex_lines.append(r"\multicolumn{3}{l}{\textbf{Panel A: Average Treatment Effect on the Treated (ATT)}} \\")
for pol, clean_name in policy_map.items():
    v = get_row_vals(pol, 'Global_ATT')
    latex_lines.append(f"\\hspace{{4mm}} {clean_name} & {v[0]} & {v[1]} \\\\")

# Panel B
latex_lines.append(r"\addlinespace")
latex_lines.append(r"\multicolumn{3}{l}{\textbf{Panel B: Group ATT (Heterogeneity by City Type)}} \\")
latex_lines.append(r"\multicolumn{3}{l}{\textit{Cluster 1: Dense Metropolis}} \\")
for pol, clean_name in policy_map.items():
    if pol == 'cp_x_lez':  # Skip Synergy for Cluster 1
        continue
    v = get_row_vals(pol, 'GATT_Cluster_1')
    latex_lines.append(f"\\hspace{{4mm}} {clean_name} & {v[0]} & {v[1]} \\\\")

latex_lines.append(r"\multicolumn{3}{l}{\textit{Cluster 0: Sprawling Regional Hub}} \\")
for pol, clean_name in policy_map.items():
    if pol == 'cp_x_lez':  # Skip Synergy for Cluster 0
        continue
    v = get_row_vals(pol, 'GATT_Cluster_0')
    latex_lines.append(f"\\hspace{{4mm}} {clean_name} & {v[0]} & {v[1]} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"") 
latex_lines.append(r"\raggedright \footnotesize \textit{Notes:} $p$-values in parentheses. $^{*} p < 0.10$, $^{**} p < 0.05$, $^{***} p < 0.01$. 'Failed' estimation bounds denote overlap violations.")
latex_lines.append(r"\end{table}")


with open(tables_dir / 'dml_master_results.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: dml_master_results.tex saved.")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# ---------------------------------------------------------
# 4. Figure 1: Coefficient Forest Plot (ATT & GATTs)
# ---------------------------------------------------------
print("Generating Figure 1: Coefficient Forest Plot (ATT & GATTs)...")

def approx_se(coef, pval):
    if pd.isna(coef) or pd.isna(pval): 
        return 0
    p = max(pval, 1e-15) 
    z = np.abs(norm.ppf(p / 2))
    return abs(coef) / z

# Map your dataframe's column names for the GATT coefficients and p-values here
col_att_coef, col_att_pval = 'Global_ATT_coef', 'Global_ATT_pval'
col_g0_coef, col_g0_pval = 'GATT_Cluster_0_coef', 'GATT_Cluster_0_pval'  # Sprawling Hubs
col_g1_coef, col_g1_pval = 'GATT_Cluster_1_coef', 'GATT_Cluster_1_pval'  # Dense Metropolises

# Calculate standard errors for all three estimands
dml_core['ATT_se'] = dml_core.apply(lambda row: approx_se(row[col_att_coef], row[col_att_pval]), axis=1)
dml_core['GATT_0_se'] = dml_core.apply(lambda row: approx_se(row[col_g0_coef], row[col_g0_pval]), axis=1)
dml_core['GATT_1_se'] = dml_core.apply(lambda row: approx_se(row[col_g1_coef], row[col_g1_pval]), axis=1)

# Increased height to comfortably fit 18 total bars (6 per policy)
fig, ax = plt.subplots(figsize=(10, 8))

y_labels = [policy_map[p] for p in core_policies]
y_ticks = np.arange(len(y_labels))

estimands = ['ATT', 'GATT_0', 'GATT_1']
# Map the display labels used in the figure to the actual model names in the source data
model_plot_map = {
    '(1) AIPW': 'L1 (Lasso / Logit L1)',
    '(2) Causal Forest': 'Causal Forest'
}
models_to_plot = list(model_plot_map.keys())

# Color mapping: Primary hue = Estimand | Shade = Algorithm
colors = {
    'ATT':    {'(1) AIPW': '#969696', '(2) Causal Forest': '#252525'}, # Greys
    'GATT_0': {'(1) AIPW': '#6baed6', '(2) Causal Forest': '#08519c'}, # Blues (Sprawling)
    'GATT_1': {'(1) AIPW': '#fb6a4a', '(2) Causal Forest': '#cb181d'}  # Reds (Dense)
}

# Vertical offsets to separate the 6 estimates neatly around the major y-tick
offsets = {
    'ATT':    {'(1) AIPW': 0.35, '(2) Causal Forest': 0.22},
    'GATT_0': {'(1) AIPW': 0.05,  '(2) Causal Forest': -0.08},
    'GATT_1': {'(1) AIPW': -0.25, '(2) Causal Forest': -0.38}
}

coef_cols = {'ATT': col_att_coef, 'GATT_0': col_g0_coef, 'GATT_1': col_g1_coef}
se_cols = {'ATT': 'ATT_se', 'GATT_0': 'GATT_0_se', 'GATT_1': 'GATT_1_se'}

handles_dict = {}

for display_mod in models_to_plot:
    actual_mod = model_plot_map[display_mod]
    subset = dml_core[dml_core['Model'] == actual_mod].set_index('Policy').reindex(core_policies)
    
    for est in estimands:
        y_pos = y_ticks + offsets[est][display_mod]
        coefs = subset[coef_cols[est]]
        ses = subset[se_cols[est]]
        
        # Plot the error bars
        eb = ax.errorbar(coefs, y_pos, xerr=ses * 1.96, 
                         fmt='o', color=colors[est][display_mod], 
                         capsize=4, elinewidth=2, markersize=7)
        
        # Save a handle for the custom legend
        clean_mod_name = display_mod.split(') ')[-1]
        label_str = f"{est} - {clean_mod_name}"
        handles_dict[label_str] = eb[0]

ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_xlabel("Treatment Effect on the Treated (Log Transport CO2)")

# Construct a clean, ordered custom legend
legend_keys = [
    'ATT - AIPW', 'ATT - Causal Forest', 
    'GATT_0 - AIPW', 'GATT_0 - Causal Forest',
    'GATT_1 - AIPW', 'GATT_1 - Causal Forest'
]

legend_labels = [k.replace('GATT_0', 'GATT (Sprawling Hubs)').replace('GATT_1', 'GATT (Dense Metropolises)') for k in legend_keys]
handles = [handles_dict[k] for k in legend_keys]

ax.legend(
    handles,
    legend_labels,
    title="Estimand & Algorithm",
    loc="lower left",
    bbox_to_anchor=(0.0, 0.0),
    frameon=True,
    framealpha=0.95,
    borderpad=0.4,
    handlelength=1.8,
    handletextpad=0.6,
)

plt.tight_layout()
fig.savefig(figures_dir / 'forest_plot_att_gatt.pdf', format='pdf', bbox_inches='tight')
plt.close()


# ---------------------------------------------------------
# 6. Figure 3: Common Support (Overlap) Density Plot
# ---------------------------------------------------------
print("Generating Figure 3: Common Support Density Plot...")

# Load the exact multinomial propensity scores
ps_df = pd.read_csv(results_dir / 'propensity_scores_exact_aipw.csv')

fig, ax = plt.subplots(figsize=(8, 5))
# UPDATED: Using 'is_cp_only' and 'propensity_score_cp_only'
sns.kdeplot(ps_df.loc[ps_df['is_cp_only'] == 1, 'propensity_score_cp_only'], fill=True, label="Treated (CP Only)", color="#d95f02", ax=ax, alpha=0.5)
sns.kdeplot(ps_df.loc[ps_df['is_cp_only'] == 0, 'propensity_score_cp_only'], fill=True, label="Untreated", color="#7570b3", ax=ax, alpha=0.5)

ax.set_xlim(0, 1)
ax.set_xlabel("Estimated Propensity Score $\\hat{P}(D=\\text{CP Only}|X)$")
ax.set_ylabel("Density")
ax.set_title("Figure 3: Common Support for Congestion Pricing", pad=15, fontweight='bold')

ax.legend()

plt.tight_layout()
fig.savefig(figures_dir / 'overlap_density.pdf', format='pdf')
plt.close()

# ---------------------------------------------------------
# Console Summary: Common Support Numerical Distribution
# ---------------------------------------------------------
# UPDATED: Using exact column names
ps_treated = ps_df.loc[ps_df['is_cp_only'] == 1, 'propensity_score_cp_only']
ps_control = ps_df.loc[ps_df['is_cp_only'] == 0, 'propensity_score_cp_only']

# 1. Summary Statistics Table
ps_summary = pd.DataFrame({
    'Treated (CP Only)': ps_treated.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]),
    'Untreated': ps_control.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
}).round(4)

# 2. Compute Common Support Boundaries
overlap_min = max(ps_treated.min(), ps_control.min())
overlap_max = min(ps_treated.max(), ps_control.max())
n_treated_in_range = (ps_treated >= overlap_min) & (ps_treated <= overlap_max)
n_control_in_range = (ps_control >= overlap_min) & (ps_control <= overlap_max)

print("\n" + "="*70)
print("NUMERICAL PROPENSITY SCORE DISTRIBUTION (CP ONLY COMMON SUPPORT)")
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
print("Exporting Comprehensive Propensity Score Summary Table to LaTeX...")

# 1. Define treatments matching exact column names in the CSV
treatments = {
    'CP': ('is_cp_only', 'propensity_score_cp_only'),
    'LEZ': ('is_lez_only', 'propensity_score_lez_only'),
    'Synergy': ('is_synergy', 'propensity_score_synergy')
}

# Add the 25% and 75% quartiles to the rows we want to extract
rows_to_keep = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
index_mapping = {
    'count': 'Observations (N)',
    'mean': 'Mean',
    'std': 'Std. Dev.',
    'min': 'Minimum',
    '25%': '25th Pctl.',
    '50%': 'Median',
    '75%': '75th Pctl.',
    'max': 'Maximum'
}

# 2. Iterate through policies and compile the summary statistics
summary_dfs = []
for short_name, (treat_col, ps_col) in treatments.items():
    if treat_col in ps_df.columns and ps_col in ps_df.columns:
        # Calculate stats for treated and untreated separately
        treated_ps = ps_df.loc[ps_df[treat_col] == 1, ps_col].describe()
        control_ps = ps_df.loc[ps_df[treat_col] == 0, ps_col].describe()
        
        # Combine into a temporary dataframe
        mini_df = pd.DataFrame({
            f"{short_name}_T": treated_ps,
            f"{short_name}_U": control_ps
        }).loc[rows_to_keep]
        summary_dfs.append(mini_df)

# Concatenate all policies horizontally into one master dataframe
ps_summary_all = pd.concat(summary_dfs, axis=1)
ps_summary_all = ps_summary_all.rename(index=index_mapping)

# 3. Build manual LaTeX lines
latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Propensity Score Distributions Across All Policy Regimes}")
latex_lines.append(r"\label{tab:propensity_scores_all}")

# Generate column format dynamically (1 left-aligned for stats + 6 centered for data)
latex_lines.append(r"\begin{tabular}{lcccccc}")
latex_lines.append(r"\toprule")

# Top header with multicolumn spans grouping the Treated/Untreated by policy
latex_lines.append(r"& \multicolumn{2}{c}{Congestion Pricing} & \multicolumn{2}{c}{Low Emission Zone} & \multicolumn{2}{c}{Synergy (CP $\times$ LEZ)} \\")
latex_lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
latex_lines.append(r"Statistic & Treated & Untreated & Treated & Untreated & Treated & Untreated \\")
latex_lines.append(r"\midrule")

# Populate data rows
for index, row in ps_summary_all.iterrows():
    row_str = f"{index}"
    for col_name in ps_summary_all.columns:
        val = row[col_name]
        if 'Observations' in index:
            row_str += f" & {int(val)}"
        else:
            row_str += f" & {val:.3f}"
    row_str += r" \\"
    latex_lines.append(row_str)

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")

# 4. Add the formatted footnote referencing the support table
latex_lines.append(r"") 
latex_lines.append(r"\vspace{1ex}")
latex_lines.append(r"{\raggedright \footnotesize \textit{Notes:} Table displays the descriptive statistics of the estimated propensity scores prior to overlap trimming. ``Untreated'' refers strictly to baseline ($D=0$) units. See Table \ref{tab:unified_common_support} for policy-specific common support bounds and sample retention rates.\par}")
latex_lines.append(r"\end{table}")

# 5. Save file to disk
with open(tables_dir / 'propensity_score_summary_all.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: propensity_score_summary_all.tex saved.")

# ---------------------------------------------------------
# Export LaTeX Table: Unified Common Support Across All Policies
# ---------------------------------------------------------
print("Generating Unified Common Support Table...")

# Define treatments matching exact column names in the new CSV
treatments = {
    'Congestion Pricing (CP Only)': ('is_cp_only', 'propensity_score_cp_only'),
    'Low Emission Zone (LEZ Only)': ('is_lez_only', 'propensity_score_lez_only'),
    'Synergy (CP $\\times$ LEZ)':   ('is_synergy', 'propensity_score_synergy')
}

latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Common Support Bounds and Sample Retention by Policy}")
latex_lines.append(r"\label{tab:unified_common_support}")
latex_lines.append(r"\begin{tabular}{lcccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Policy & Support Region & Treated Retained & Untreated Retained \\")
latex_lines.append(r"\midrule")

for name, (treat_col, ps_col) in treatments.items():
    if treat_col in ps_df.columns and ps_col in ps_df.columns:
        ps_treated_uni = ps_df.loc[ps_df[treat_col] == 1, ps_col]
        ps_control_uni = ps_df.loc[ps_df[treat_col] == 0, ps_col]
        
        overlap_min_uni = max(ps_treated_uni.min(), ps_control_uni.min())
        overlap_max_uni = min(ps_treated_uni.max(), ps_control_uni.max())
        
        n_treated_uni = len(ps_treated_uni)
        n_control_uni = len(ps_control_uni)
        treated_in_uni = ((ps_treated_uni >= overlap_min_uni) & (ps_treated_uni <= overlap_max_uni)).sum()
        control_in_uni = ((ps_control_uni >= overlap_min_uni) & (ps_control_uni <= overlap_max_uni)).sum()
        
        treated_pct_uni = (treated_in_uni / n_treated_uni) * 100 if n_treated_uni > 0 else 0
        control_pct_uni = (control_in_uni / n_control_uni) * 100 if n_control_uni > 0 else 0
        
        bounds_str = f"$[{overlap_min_uni:.3f}, {overlap_max_uni:.3f}]$"
        treated_str = f"{treated_in_uni} ({treated_pct_uni:.1f}\\%)"
        control_str = f"{control_in_uni} ({control_pct_uni:.1f}\\%)"
        
        latex_lines.append(f"{name} & {bounds_str} & {treated_str} & {control_str} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\vspace{1ex}")
latex_lines.append(r"{\raggedright \footnotesize \textit{Notes:} The common support region is defined mathematically as $[\max(\min_T, \min_C), \min(\max_T, \max_C)]$ for each respective policy's propensity score distribution. Units outside these bounds are strictly trimmed prior to causal estimation to ensure finite-sample overlap.\par}")
latex_lines.append(r"\end{table}")

with open(tables_dir / 'unified_common_support.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: unified_common_support.tex saved.")
print(f"Success: All figures generated and saved as PDFs in {figures_dir}")