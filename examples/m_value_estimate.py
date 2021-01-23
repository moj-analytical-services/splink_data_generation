from generate_data_random import generate_df_gammas_random

from estimate_splink import estimate
import statsmodels.api as sm

settings = {
    "proportion_of_matches": 0.3,
    "link_type": "dedupe_only",
    "comparison_columns": [
        {
            "col_name": "first_name",
            "num_levels": 3,
            "m_probabilities": [0.05, 0.2, 0.75],
            "u_probabilities": [0.9, 0.05, 0.05],
        },
        {
            "col_name": "surname",
            "m_probabilities": [0.02, 0.98],
            "u_probabilities": [0.95, 0.05],
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

# We expect independence conditional on match status
df_gammas = generate_df_gammas_random(1e6, settings)

# f1 = df_gammas["gamma_surname"] == 1
# df_gammas = df_gammas[f1]

# settings = {
#     "proportion_of_matches": 0.3,
#     "link_type": "dedupe_only",
#     "comparison_columns": [
#         {
#             "col_name": "first_name",
#             "num_levels": 3,
#         },
#         {
#             "col_name": "col_3",
#         },
#     ],
#     "retain_matching_columns": False,
#     "retain_intermediate_calculation_columns": False,
# }


# df_e, linker = estimate(df_gammas, settings)

# print(linker.model)

# linker.model.bayes_factor_chart()

# We have the real answer so use that to compute m and u
f1 = df_gammas["true_match_l"] == 1

df_gammas_matches = df_gammas[f1]
df_gammas_matches.groupby("gamma_surname")["unique_id_l"].count() / len(
    df_gammas_matches
)

f2 = df_gammas_matches["gamma_col_3"] == 1
df_gamma_matches_col_3_match = df_gammas_matches[f2]

df_gamma_matches_col_3_match.groupby("gamma_surname")["unique_id_l"].count() / len(
    df_gamma_matches_col_3_match
)
