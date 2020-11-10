from generate_data import generate_df_gammas
from estimate_splink import estimate

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

df_gammas = generate_df_gammas(500000, settings)


dfe, linker = estimate(df_gammas, settings)

print(linker.params)

linker.params.bayes_factor_chart()