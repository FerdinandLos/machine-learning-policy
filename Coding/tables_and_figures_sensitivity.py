import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
results_dir = Path('Data/Results')
tables_dir = Path('Writing/Tables')
figures_dir = Path('Writing/Figures')

tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("Loading sensitivity datasets...")
try:
    placebo_df = pd.read_csv(results_dir / 'sensitivity_placebo_tests_combined.csv')
    overlap_df = pd.read_csv(results_dir / 'sensitivity_overlap_trimming.csv')
except FileNotFoundError as e:
    print(f" [!] Missing data file: {e}. Please ensure the sensitivity scripts have been run.")
    exit()

policy_map = {
    'cp_active': 'Congestion Pricing (CP)',
    'lez_active': 'Low Emission Zone (LEZ)',
    'cp_x_lez': 'Synergy (CP x LEZ)'
}

outcome_map = {
    'museum_visitors_pc': 'No. of Museum Visitors (pc)', 
    'library_count': 'No. of Libraries', 
    'streetlight_density': 'Streetlight Density', 
    'fountain_count': 'No. of Fountains', 
    'bench_count_pc': 'No. of Benches pc', 
    'flagpole_count': 'No. of flagpoles', 
    'sister_city_count': 'No. of Sister Cities'
}

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

 # ---------------------------------------------------------
# 2. Process and Export Placebo Falsification Table
# ---------------------------------------------------------

print("Formatting Placebo Falsification Table...")

# Ensure your translation mappings are explicitly declared
policy_map = {
    'cp_active': 'Congestion Pricing (CP)',
    'lez_active': 'Low Emission Zone (LEZ)',
    'cp_x_lez': 'Synergy (CP $\\times$ LEZ)'
}

# Ensure the format function adds LaTeX math mode markers ($) around stars

def format_estimate(coef, pval):
    if pd.isna(coef) or pd.isna(pval):

        return "--"
    stars = ""
    if pval < 0.01: stars = "^{***}"
    elif pval < 0.05: stars = "^{**}"
    elif pval < 0.10: stars = "^{*}"
    return f"${coef:.4f}{stars}$ ($p={pval:.3f}$)"

# Apply formatting directly
placebo_df['Global_ATT_fmt'] = placebo_df.apply(
    lambda row: format_estimate(row['Global_ATT_coef'], row['Global_ATT_pval']), axis=1
)
# Extract your unique outcomes present in the dataframe
outcome_list = placebo_df['Outcome'].unique()

# Safer value extraction function that matches substrings to protect against typos
def get_placebo_val(outcome, model_substring, policy):
    # Filter by outcome and policy first
    sub_df = placebo_df[(placebo_df['Outcome'] == outcome) & (placebo_df['Policy'] == policy)]
    # Look for model string safely (e.g. contains 'L1' or 'Lasso' or 'Causal Forest')
    match = sub_df[sub_df['Model'].str.contains(model_substring, case=False, na=False)]['Global_ATT_fmt'].values
    return match[0] if len(match) > 0 else "--"


# Build clean manual LaTeX table lines to avoid pandas MultiIndex engine crashes
latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Falsification Tests: Estimated Effects on Placebo Outcomes (Global ATT)}")
latex_lines.append(r"\label{tab:placebo_tests}")
# Reduced to 4 columns: 1 for labels, 3 for the policies
latex_lines.append(r"\begin{tabular}{lccc}") 
latex_lines.append(r"\toprule")

# Header Row: Policy Titles
p1 = policy_map['cp_active']
p2 = policy_map['lez_active']
p3 = policy_map['cp_x_lez']
latex_lines.append(f"Decoy Outcome & {p1} & {p2} & {p3} \\\\")

# ---------------------------------------------------------
# Panel A: AIPW (Lasso)
# ---------------------------------------------------------
latex_lines.append(r"\midrule")
latex_lines.append(r"\multicolumn{4}{l}{\textbf{Panel A: AIPW (Lasso)}} \\")
latex_lines.append(r"\midrule")

for out in outcome_list:
    clean_outcome_name = outcome_map.get(out, out) if 'outcome_map' in locals() or 'outcome_map' in globals() else out
    
    v1 = get_placebo_val(out, 'L1', 'cp_active')
    v2 = get_placebo_val(out, 'L1', 'lez_active')
    v3 = get_placebo_val(out, 'L1', 'cp_x_lez')
    
    latex_lines.append(f"{clean_outcome_name} & {v1} & {v2} & {v3} \\\\")

# ---------------------------------------------------------
# Panel B: Causal Forest
# ---------------------------------------------------------
latex_lines.append(r"\midrule")
latex_lines.append(r"\multicolumn{4}{l}{\textbf{Panel B: Causal Forest}} \\")
latex_lines.append(r"\midrule")

for out in outcome_list:
    clean_outcome_name = outcome_map.get(out, out) if 'outcome_map' in locals() or 'outcome_map' in globals() else out
    
    v4 = get_placebo_val(out, 'Causal', 'cp_active')
    v5 = get_placebo_val(out, 'Causal', 'lez_active')
    v6 = get_placebo_val(out, 'Causal', 'cp_x_lez')
    
    latex_lines.append(f"{clean_outcome_name} & {v4} & {v5} & {v6} \\\\")

# Finalize table formatting
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\raggedright \footnotesize \textit{Notes:} $p$-values in parentheses. $^{*}$ $p < 0.10$, $^{**}$ $p < 0.05$, $^{***}$ $p < 0.01$.")
latex_lines.append(r"\end{table}")
0
# Save file to disk
with open(tables_dir / 'appendix_placebo_tests.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: appendix_placebo_tests.tex saved with vertical panel stacking.")

# ---------------------------------------------------------
# 3. Figure: Sensitivity to Overlap Trimming (with 95% CIs)
# ---------------------------------------------------------
print("Generating Overlap Trimming Sensitivity Plot with Confidence Intervals...")

from scipy.stats import norm
import numpy as np

# 1. Load the main estimation results to get the 0.01 threshold baseline
main_results_path = results_dir / 'dml_robustness_results_aipw.csv'
main_df = pd.read_csv(main_results_path)

# Filter strictly for the Lasso/AIPW model
main_l1 = main_df[main_df['Model'] == 'L1 (Lasso / Logit L1)'].copy()

# Add the explicit trimming threshold used in the main script
main_l1['Trim_Threshold'] = 0.01

# 2. Safety Filter for the sensitivity dataframe
if 'Model' in overlap_df.columns:
    overlap_df = overlap_df[overlap_df['Model'] == 'L1 (Lasso / Logit L1)']

# 3. Merge the main 0.01 estimates into the sensitivity dataframe
combined_df = pd.concat([overlap_df, main_l1], ignore_index=True)
combined_df['Trim_Threshold'] = pd.to_numeric(combined_df['Trim_Threshold'])

# --- NEW: Reverse-Engineer Standard Errors from p-values ---
# Clip the p-value to avoid division by zero in the rare case where p is exactly 1.0
clipped_pval = np.clip(combined_df['Global_ATT_pval'], 1e-10, 0.99999)

# Calculate the Z-statistic and derive the Standard Error
combined_df['z_stat'] = np.abs(norm.ppf(clipped_pval / 2))
combined_df['se'] = np.abs(combined_df['Global_ATT_coef']) / combined_df['z_stat']

# Calculate the 95% Confidence Interval margin (+/- 1.96 * SE)
combined_df['ci_margin'] = 1.96 * combined_df['se']

# 4. Global Plot Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(10, 6))

colors = {'cp_active': '#d95f02', 'lez_active': '#7570b3', 'cp_x_lez': '#1b9e77'}
markers = {'cp_active': 'o', 'lez_active': 's', 'cp_x_lez': '^'}

for policy_key, policy_label in policy_map.items():
    # Sort mathematically first to ensure 0.01 lands right between 0.005 and 0.05
    subset = combined_df[combined_df['Policy'] == policy_key].sort_values('Trim_Threshold')
    
    # We plot the Threshold as a string/categorical so the x-axis spacing is even
    x_vals = subset['Trim_Threshold'].astype(str)
    
    # Replace standard plot with errorbar to include 95% CI bands
    ax.errorbar(x_vals, subset['Global_ATT_coef'], yerr=subset['ci_margin'], 
                fmt=markers[policy_key] + '-', # Line with marker
                linewidth=2, markersize=8, 
                capsize=5, capthick=1.5, elinewidth=1.5, # Cap styling for the error bars
                color=colors[policy_key], label=policy_label, alpha=0.9)

ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

ax.set_xlabel("Propensity Score Trimming Threshold (min_propensity)", labelpad=10)
ax.set_ylabel("Global ATT Estimate (Log Transport $CO_2$)", labelpad=10)
ax.legend(title="Policy Regime", loc="best")

plt.tight_layout()
fig.savefig(figures_dir / 'appendix_overlap_sensitivity.pdf', format='pdf', bbox_inches='tight')
plt.close()

print(f"Success: Sensitivity figures with 95% CIs generated and saved to {figures_dir}")