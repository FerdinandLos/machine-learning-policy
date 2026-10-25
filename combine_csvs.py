import pandas as pd

df1 = pd.read_csv("Data/Results/sensitivity_placebo_tests.csv")
df2 = pd.read_csv("Data/Results/sensitivity_placebo_tests2.csv")

combined = pd.concat([df1, df2], ignore_index=True)

combined.to_csv("Data/Results/sensitivity_placebo_tests_combined.csv", index=False)

print("Merged rows:", len(combined))