# https://imai.fas.harvard.edu/research/files/linkage.pdf
# See page 356
# There is an assumption of "conditional independence among linkage variables"
# given the match status

from splink_data_generation.generate_data_exact import generate_df_gammas_exact
from splink_data_generation.generate_data_random import generate_df_gammas_random
import numpy as np
import pytest
from tests.settings import settings_1, settings_2


@pytest.mark.parametrize(
    "settings",
    [settings_1, settings_2],
)
def test_conditional_independence_exact(settings):

    df_gammas = generate_df_gammas_exact(settings)

    gamma_cols = [c for c in df_gammas.columns if "gamma_" in c]

    f1 = df_gammas["true_match_l"] == 0

    df_gammas_match = df_gammas[f1]

    result = df_gammas_match[gamma_cols].corr().to_numpy()
    expected = np.identity(len(gamma_cols))

    np.testing.assert_allclose(result, expected, atol=1e-10)

    f2 = df_gammas["true_match_l"] == 1

    df_gammas_non_match = df_gammas[f2]

    result = df_gammas_non_match[gamma_cols].corr().to_numpy()
    expected = np.identity(len(gamma_cols))

    np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.parametrize(
    "settings",
    [settings_1, settings_2],
)
def test_conditional_independence_random(settings):

    np.random.seed(14)

    df_gammas = generate_df_gammas_random(20000, settings)

    gamma_cols = [c for c in df_gammas.columns if "gamma_" in c]

    f1 = df_gammas["true_match_l"] == 0

    df_gammas_match = df_gammas[f1]

    result = df_gammas_match[gamma_cols].corr().to_numpy()
    expected = np.identity(len(gamma_cols))

    np.testing.assert_allclose(result, expected, atol=0.05)

    f2 = df_gammas["true_match_l"] == 1

    df_gammas_non_match = df_gammas[f2]

    result = df_gammas_non_match[gamma_cols].corr().to_numpy()
    expected = np.identity(len(gamma_cols))

    np.testing.assert_allclose(result, expected, atol=0.05)
