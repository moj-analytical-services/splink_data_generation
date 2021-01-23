from generate_data_random import generate_df_gammas_random
from estimate_splink import estimate
import statsmodels.api as sm
import numpy as np

settings = {
    "proportion_of_matches": 0.3,
    "link_type": "dedupe_only",
    "comparison_columns": [
        {
            "col_name": "first_name",
            "num_levels": 3,
            "m_probabilities": [0.05, 0.2, 0.75],
            "u_probabilities": [0.9, 0.08, 0.02],
        },
        {
            "col_name": "surname",
            "m_probabilities": [0.02, 0.98],
            "u_probabilities": [0.99, 0.01],
        },
        {
            "col_name": "col_3",
            "m_probabilities": [0.1, 0.9],
            "u_probabilities": [0.8, 0.2],
        },
    ],
    "retain_matching_columns": False,
    "retain_intermediate_calculation_columns": False,
}

# Create covariance between first_name and surname
df_gammas = generate_df_gammas_random(
    200000, settings, corr_overrides=[["first_name", "surname", 0.95]]
)

df_match = df_gammas[df_gammas["true_match_l"] == 0]
y = df_match["gamma_first_name"]
X = df_match[["gamma_surname", "gamma_col_3"]]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


df_e, linker = estimate(df_gammas, settings)

print(linker.model)

linker.model.bayes_factor_chart()

# What happens if we use the estimated match probability to subset matches and non-matches?
df_e_pd = df_e.toPandas()
df_e_pd["match"] = np.where(df_e_pd["match_probability"] > 0.5, 1, 0)
f1 = df_e_pd["match"] == 1
df_e_pd_m = df_e_pd[f1]

y = df_e_pd_m["gamma_first_name"]
X = df_e_pd_m[["gamma_surname", "gamma_col_3"]]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
