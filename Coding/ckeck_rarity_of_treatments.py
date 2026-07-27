import pandas as pd

# Load the dataset
df = pd.read_csv('Data/urban_emissions_panel_cleaned.csv')

# Ensure the synergy variable exists
if 'cp_x_lez' not in df.columns:
    df['cp_x_lez'] = df['cp_active'] * df['lez_active']

total_obs = len(df)

print(f"--- TREATMENT RARITY ANALYSIS (Total Observations: {total_obs}) ---")

for policy in ['cp_active', 'lez_active', 'cp_x_lez']:
    count = df[policy].sum()
    percentage = (count / total_obs) * 100
    print(f"{policy.upper()}:")
    print(f"  Count: {count} active city-years")
    print(f"  Prevalence: {percentage:.2f}% of total sample\n")