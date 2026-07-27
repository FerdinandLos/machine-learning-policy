import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from econml.dr import LinearDRLearner

root = Path(r'c:/Users/ferdi/Documents/Study Projects/Machine Learning/machine-learning-policy')
df = pd.read_csv(root/'Data/urban_emissions_panel_cleaned.csv')
df['cp_x_lez'] = df['cp_active'] * df['lez_active']
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=int)
df = pd.concat([df, year_dummies], axis=1)
exclude_from_W = ['city_id','year','log_transport_co2','log_total_co2','cp_active','lez_active','cp_impl_year','lez_impl_year','cp_announce_year','lez_announce_year','country_id','cp_x_lez','industry_public','fleet_petrol_share']
base_W_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_from_W]
cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='Cluster', dtype=int)
cluster_dummies_array = cluster_dummies.to_numpy()
W_matrix = df[base_W_cols + ['lez_active']].to_numpy()

ml_l = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1))
ml_m = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, n_jobs=-1))
cv_panel = GroupKFold(n_splits=5)
estimator = LinearDRLearner(model_regression=clone(ml_l), model_propensity=clone(ml_m), min_propensity=0.01, cv=cv_panel, random_state=42, fit_cate_intercept=False)

estimator.fit(
    Y=df['log_transport_co2'].to_numpy(),
    T=df['cp_active'].to_numpy(),
    X=cluster_dummies_array,
    W=W_matrix,
    groups=df['city_id'].to_numpy(),
)
print('fit ok')
ate = estimator.ate(X=cluster_dummies_array)
print('ate', ate)
