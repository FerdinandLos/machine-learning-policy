beta_1aipw <- -0.27
exact_pct_change_1aipw <- (exp(beta_1aipw) - 1) * 100
beta_1cf <- -0.37
exact_pct_change_1cf <- (exp(beta_1cf) - 1) * 100
print("CP_aipw=")
print(exact_pct_change_1aipw)
print("CP_cf=")
print(exact_pct_change_1cf)

beta_2aipw <- -0.11
exact_pct_change_2aipw <- (exp(beta_2aipw) - 1) * 100
beta_2cf <- -0.16
exact_pct_change_2cf <- (exp(beta_2cf) - 1) * 100
print("LEZ_aipw=")
print(exact_pct_change_2aipw)
print("LEZ_cf=")
print(exact_pct_change_2cf)

# Added together

exact_pct_change_comb_aipw <- (exp(beta_1aipw + beta_2aipw) - 1) * 100
exact_pct_change_comb_cf <- (exp(beta_1cf +beta_2cf) - 1) * 100
print("Comb_aipw=")
print(exact_pct_change_comb_aipw)
print("Comb_cp=")
print(exact_pct_change_comb_cf)

beta_3 <- -1.25
exact_pct_change_3 <- (exp(beta_3) - 1) * 100
print("Synergy=")
print(exact_pct_change_3)
