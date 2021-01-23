from splink_data_generation.generate_data_exact import generate_df_gammas_exact
from splink_data_generation.match_prob import add_match_prob
from splink_data_generation.log_likelihood import add_log_likelihood

import pandas as pd
import pytest

# Answers generated using
# from splink import Splink
# from splink.expectation_step import _calculate_log_likelihood_df
# from splink.model import Model
# linker = Splink(settings, df=df, spark=spark)
# df_e = linker.manually_apply_fellegi_sunter_weights()
# model = Model(settings, spark)
# cols = ["match_probability", "log_likelihood", "gamma_col_1", "gamma_col_2", "gamma_col_3"]
# df_pd = _calculate_log_likelihood_df(df_e, model, spark).toPandas()
# df_pd[cols].to_dict(orient="records")


def test_ll_match_prob():

    settings = {
        "proportion_of_matches": 0.5,
        "link_type": "dedupe_only",
        "comparison_columns": [
            {
                "col_name": "col_1",
                "m_probabilities": [0.3, 0.7],  # Probability of typo
                "u_probabilities": [0.9, 0.1],  # Probability of collision
            },
            {
                "col_name": "col_2",
                "m_probabilities": [0.1, 0.9],  # Probability of typo
                "u_probabilities": [0.975, 0.025],  # Probability of collision
            },
            {
                "col_name": "col_3",
                "m_probabilities": [0.05, 0.95],  # Probability of typo
                "u_probabilities": [0.8, 0.2],  # Probability of collision
            },
        ],
        "max_iterations": 0,
        "em_convergence": 0.0001,
    }

    expected_result_rows = [
        {
            "match_probability": 0.9402985074626865,
            "log_likelihood": -4.089357020711062,
            "gamma_col_1": 1,
            "gamma_col_2": 1,
            "gamma_col_3": 0,
        },
        {
            "match_probability": 0.0021321961620469087,
            "log_likelihood": -1.0448345829876389,
            "gamma_col_1": 0,
            "gamma_col_2": 0,
            "gamma_col_3": 0,
        },
        {
            "match_probability": 0.04294478527607361,
            "log_likelihood": -3.2002994392952653,
            "gamma_col_1": 1,
            "gamma_col_2": 0,
            "gamma_col_3": 0,
        },
        {
            "match_probability": 0.0021321961620469087,
            "log_likelihood": -1.0448345829876389,
            "gamma_col_1": 0,
            "gamma_col_2": 0,
            "gamma_col_3": 0,
        },
        {
            "match_probability": 0.04294478527607361,
            "log_likelihood": -3.2002994392952653,
            "gamma_col_1": 1,
            "gamma_col_2": 0,
            "gamma_col_3": 0,
        },
        {
            "match_probability": 0.9827586206896551,
            "log_likelihood": -2.036382052219389,
            "gamma_col_1": 0,
            "gamma_col_2": 1,
            "gamma_col_3": 1,
        },
    ]
    df_expected = pd.DataFrame(expected_result_rows)

    df = generate_df_gammas_exact(settings)
    df = add_match_prob(df, settings)
    df = add_log_likelihood(df, settings)
    # df = add_log_likelihood_vectorised(df, settings)

    join = ["gamma_col_1", "gamma_col_2", "gamma_col_3"]
    merged = df.merge(
        df_expected, left_on=join, right_on=join, how="inner"
    ).drop_duplicates(join)

    assert list(merged["true_match_probability_l"]) == pytest.approx(
        list(merged["match_probability"])
    )
    assert list(merged["true_log_likelihood_l"]) == pytest.approx(
        list(merged["log_likelihood"])
    )

    assert list(merged["true_match_probability_r"]) == pytest.approx(
        list(merged["match_probability"])
    )
    assert list(merged["true_log_likelihood_r"]) == pytest.approx(
        list(merged["log_likelihood"])
    )
