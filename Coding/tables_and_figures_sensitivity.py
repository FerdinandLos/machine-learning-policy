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
    placebo_df = pd.read_csv(results_dir / 'sensitivity_placebo_tests.csv')
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
    'museum_visitors_pc': 'Museum Visitors (per cap)', 
    'library_count': 'Library Count', 
    'streetlight_density': 'Streetlight Density', 
    'fountain_count': 'Fountain Count', 
    'bench_count_pc': 'Bench Count (per cap)', 
    'flagpole_count': 'Flagpole Count', 
    'sister_city_count': 'Sister City Count'
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
latex_lines.append(r"\begin{tabular}{lcccccc}")
latex_lines.append(r"\toprule")

# Header Row 1: Explicit 3-column span logic for the 7-track grid
latex_lines.append(r"Model & \multicolumn{3}{c}{AIPW (Lasso)} & \multicolumn{3}{c}{Causal Forest} \\")
latex_lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")

# Header Row 2: Policy Titles with proper LaTeX formatting
p1 = policy_map['cp_active']
p2 = policy_map['lez_active']
p3 = policy_map['cp_x_lez']
latex_lines.append(f"Policy & {p1} & {p2} & {p3} & {p1} & {p2} & {p3} \\\\")
latex_lines.append(r"Decoy Outcome &  &  &  &  &  &  \\")
latex_lines.append(r"\midrule")

# Populate data rows
for out in outcome_list:
    # Use outcome_map conversion if it exists, otherwise fall back to raw string name
    clean_outcome_name = outcome_map.get(out, out) if 'outcome_map' in locals() or 'outcome_map' in globals() else out
    
    # Extract row values using loose substring matching for models to ensure data hits
    v1 = get_placebo_val(out, 'L1', 'cp_active')
    v2 = get_placebo_val(out, 'L1', 'lez_active')
    v3 = get_placebo_val(out, 'L1', 'cp_x_lez')
    
    v4 = get_placebo_val(out, 'Causal', 'cp_active')
    v5 = get_placebo_val(out, 'Causal', 'lez_active')
    v6 = get_placebo_val(out, 'Causal', 'cp_x_lez')
    
    latex_lines.append(f"{clean_outcome_name} & {v1} & {v2} & {v3} & {v4} & {v5} & {v6} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")
latex_lines.append(r"\raggedright \footnotesize \textit{Notes:} $p$-values in parentheses. $^{*}$ $p < 0.10$, $^{**}$ $p < 0.05$, $^{***}$ $p < 0.01$.")

# Save file to disk
with open(tables_dir / 'appendix_placebo_tests.tex', 'w') as f:
    f.write("\n".join(latex_lines) + "\n")

print("Success: appendix_placebo_tests.tex saved.")

# ---------------------------------------------------------
# 3. Figure: Sensitivity to Overlap Trimming
# ---------------------------------------------------------
print("Generating Overlap Trimming Sensitivity Plot...")

# Safety Filter: Ensure we only plot Lasso, as Causal Forests do not use IPW trimming
if 'Model' in overlap_df.columns:
    overlap_df = overlap_df[overlap_df['Model'] == 'L1 (Lasso / Logit L1)']

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(9, 6))

colors = {'cp_active': '#d95f02', 'lez_active': '#7570b3', 'cp_x_lez': '#1b9e77'}
markers = {'cp_active': 'o', 'lez_active': 's', 'cp_x_lez': '^'}

for policy_key, policy_label in policy_map.items():
    subset = overlap_df[overlap_df['Policy'] == policy_key].sort_values('Trim_Threshold')
    
    # We plot the Threshold as a string/categorical so the x-axis spacing is even
    x_vals = subset['Trim_Threshold'].astype(str)
    
    ax.plot(x_vals, subset['Global_ATT_coef'], marker=markers[policy_key], 
            linewidth=2, markersize=8, color=colors[policy_key], label=policy_label)

ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

ax.set_xlabel("Propensity Score Trimming Threshold (min_propensity)", labelpad=10)
ax.set_ylabel("Global ATT Estimate (Log Transport CO2)", labelpad=10)
ax.set_title("Appendix Figure A1: Sensitivity to Positivity Trimming (AIPW Lasso)", pad=15, fontweight='bold')
ax.legend(title="Policy Regime", loc="best")

plt.tight_layout()
fig.savefig(figures_dir / 'appendix_overlap_sensitivity.pdf', format='pdf', bbox_inches='tight')
plt.close()

print(f"Success: Sensitivity figures generated and saved to {figures_dir}")