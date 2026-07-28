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

# Format the Global ATT coefficients and p-values
placebo_df['Global_ATT_fmt'] = placebo_df.apply(
    lambda row: format_estimate(row['Global_ATT_coef'], row['Global_ATT_pval']), axis=1
)

# Unstack the data to create a MultiIndex table: Models on top, Policies underneath
target_models = ['L1 (Lasso / Logit L1)', 'Causal Forest']
core_policies = ['cp_active', 'lez_active', 'cp_x_lez']

pivot_placebo = placebo_df.set_index(['Outcome', 'Model', 'Policy'])['Global_ATT_fmt'].unstack(['Model', 'Policy'])

# Enforce the correct column ordering
idx = pd.MultiIndex.from_product([target_models, core_policies], names=['Model', 'Policy'])
pivot_placebo = pivot_placebo.reindex(columns=idx)

# Rename the headers for publication
pivot_placebo.rename(columns={'L1 (Lasso / Logit L1)': 'AIPW (Lasso)', 'Causal Forest': 'Causal Forest'}, level=0, inplace=True)
pivot_placebo.rename(columns=policy_map, level=1, inplace=True)

# Rename the row indices
pivot_placebo.index = [outcome_map.get(idx, idx) for idx in pivot_placebo.index]
pivot_placebo.index.name = 'Decoy Outcome'

latex_table_placebo = pivot_placebo.to_latex(
    escape=False,
    column_format="l" + "c" * len(pivot_placebo.columns),
    caption="Falsification Tests: Estimated Effects on Placebo Outcomes (Global ATT)",
    label="tab:placebo_tests",
    multicolumn=True,
    multirow=True
)

with open(tables_dir / 'appendix_placebo_tests.tex', 'w') as f:
    f.write(latex_table_placebo)
    f.write("\\raggedright \\footnotesize \\textit{Notes:} $p$-values in parentheses. $^{*}$ $p < 0.10$, $^{**}$ $p < 0.05$, $^{***}$ $p < 0.01$.\n")

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