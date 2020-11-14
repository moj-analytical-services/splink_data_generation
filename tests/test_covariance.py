# https://imai.fas.harvard.edu/research/files/linkage.pdf
# See page 356
# There is an assumption of "conditional independence among linkage variables"
# given the match status

from splink_data_generation.generate_data_exact import generate_df_gammas_exact
import numpy as np


def test_conditional_independence_exact(simple_settings):

    df_gammas = generate_df_gammas_exact(simple_settings)

    gamma_cols = [c for c in df_gammas.columns if "gamma_" in c]

    f1 = df_gammas["true_match"] == 0

    df_gammas_match = df_gammas[f1]

    result = df_gammas_match[gamma_cols].corr().to_numpy()
    expected = np.identity(len(gamma_cols))

    assert (result == expected).all()

    f2 = df_gammas["true_match"] == 1

    df_gammas_non_match = df_gammas[f2]

    print(df_gammas_non_match)

    result = df_gammas_non_match[gamma_cols].corr().to_numpy()
    expected = np.identity(len(gamma_cols))

    np.testing.assert_allclose(result, expected, atol=1e-10)
