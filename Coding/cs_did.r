# ---------------------------------------------------------
# 1. System Setup & Data Loading
# ---------------------------------------------------------
library(did)
library(dplyr)
library(readr)
library(ggplot2)

# Ensure the figures directory exists
dir.create("Writing/Figures", recursive = TRUE, showWarnings = FALSE)

print("Loading data for Callaway & Sant'Anna DiD...")
df <- read_csv("Data/urban_emissions_panel_cleaned.csv")

# CS-DiD requires the group variable (adoption year) to be 0 for never-treated units
df <- df %>%
  mutate(
    cp_group = ifelse(is.na(cp_impl_year), 0, cp_impl_year),
    lez_group = ifelse(is.na(lez_impl_year), 0, lez_impl_year)
  )

# ---------------------------------------------------------
# 2. Covariate Definition
# ---------------------------------------------------------
# Identify the covariates (must be time-invariant or handled by csdid base period)
exclude_from_W <- c(
  "city_id", "year", "log_transport_co2", "log_total_co2",
  "cp_active", "lez_active", "cp_impl_year", "lez_impl_year",
  "cp_announce_year", "lez_announce_year", "country_id",
  "cp_x_lez", "policy_regime", "industry_public", "fleet_petrol_share",
  "cp_group", "lez_group", "cluster_id" 
)

# Get numeric columns only, then exclude the list to define base W
numeric_cols <- names(df)[sapply(df, is.numeric)]
base_W_cols <- setdiff(numeric_cols, exclude_from_W)

# Construct the exact formula expected by the 'did' package
xformla_str <- paste("~", paste(base_W_cols, collapse = " + "))
xformla <- as.formula(xformla_str)

print(paste("Formula configured with", length(base_W_cols), "baseline covariates."))

# ---------------------------------------------------------
# 3. Callaway & Sant'Anna: Congestion Pricing (CP)
# ---------------------------------------------------------
print("--- ESTIMATING CS-DID FOR CONGESTION PRICING ---")

# Step 3a: Estimate all 2x2 ATT(g,t) using Doubly Robust estimation
out_cp <- att_gt(
  yname = "log_transport_co2",
  gname = "cp_group",
  idname = "city_id",
  tname = "year",
  xformla = xformla,
  data = df,
  est_method = "dr",
  control_group = "notyettreated", # Uses clean controls only
  panel = TRUE                     # Exploits the true panel structure
)

# Step 3b: Aggregate into an Event Study (Dynamic)
print(">> Aggregating to Dynamic Event Study (CP)...")
es_cp <- aggte(out_cp, type = "dynamic")

# Step 3c: Plot and export
p_cp <- ggdid(es_cp) +
  ggtitle("Event Study: Impact of Congestion Pricing (CS-DiD)") +
  xlab("Years relative to CP Implementation") +
  ylab("ATT on Log Transport CO2") +
  theme_bw(base_family = "serif", base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("Writing/Figures/event_study_cp_csdid.pdf", plot = p_cp, width = 10, height = 6)
print("Success: Event study plot saved to Writing/Figures/event_study_cp_csdid.pdf")

# ---------------------------------------------------------
# 4. Callaway & Sant'Anna: Low Emission Zones (LEZ)
# ---------------------------------------------------------
print("--- ESTIMATING CS-DID FOR LOW EMISSION ZONES ---")

# Step 4a: Estimate all 2x2 ATT(g,t)
out_lez <- att_gt(
  yname = "log_transport_co2",
  gname = "lez_group",
  idname = "city_id",
  tname = "year",
  xformla = xformla,
  data = df,
  est_method = "dr",
  control_group = "notyettreated",
  panel = TRUE
)

# Step 4b: Aggregate into an Event Study (Dynamic)
print(">> Aggregating to Dynamic Event Study (LEZ)...")
es_lez <- aggte(out_lez, type = "dynamic")

# Step 4c: Plot and export
p_lez <- ggdid(es_lez) +
  ggtitle("Event Study: Impact of Low Emission Zones (CS-DiD)") +
  xlab("Years relative to LEZ Implementation") +
  ylab("ATT on Log Transport CO2") +
  theme_bw(base_family = "serif", base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("Writing/Figures/event_study_lez_csdid.pdf", plot = p_lez, width = 10, height = 6)
print("Success: Event study plot saved to Writing/Figures/event_study_lez_csdid.pdf")