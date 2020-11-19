# Generate synthetic datasets corresponding to user-provided
# m and u probabilities using a random number generator

# The data generating process (DGP) corresponds to the true parameters.
# Rows are generated at at random from the DGP
# so the parameters of the synthetic dataset will be close
# to the supplied m and u values.  The more rows generated
# the closer they will be

from uuid import uuid4
import math


import numpy as np
import pandas as pd
from scipy.stats import norm

from splink.settings import complete_settings_dict


def _gen_cov_matrix_no_correlation(settings):
    cc = settings["comparison_columns"]
    num = len(cc)
    a = np.zeros((num, num), float)
    np.fill_diagonal(a, 1.0)
    return a


def _generate_uniform_random_numbers(num_rows, cov):

    if type(cov) == list:
        cov = np.array(cov)
    mean = np.zeros(cov.shape[0])

    data = np.random.multivariate_normal(mean, cov, num_rows).T
    data = norm.cdf(data)
    return data


def _ran_numbers_to_gammas(data, settings, is_match):

    v = {}

    for i, col in enumerate(settings["comparison_columns"]):
        ran_vector = data[i]
        if is_match:
            bins = col["m_probabilities"][:-1]
            bins = np.cumsum(bins)
        else:
            bins = col["u_probabilities"][:-1]
            bins = np.cumsum(bins)

        col_name = col["col_name"]
        v[f"gamma_{col_name}"] = np.digitize(ran_vector, bins, right=True)
    return v


def _add_essentials_to_settings(settings):

    if "link_type" not in settings:
        settings["link_type"] == "dedupe only"
    return settings


def _get_col_index(settings, name):
    for i, col in enumerate(settings["comparison_columns"]):
        if col["col_name"] == name:
            return i


def _apply_override(override, cov, settings):
    i0 = _get_col_index(settings, override[0])
    i1 = _get_col_index(settings, override[1])

    cov[i0, i1] = override[2]
    cov[i1, i0] = override[2]
    return cov


def generate_df_gammas_random(
    num_rows: int,
    settings: dict,
    cov_m: np.ndarray = None,
    cov_overrides_m: list = [],
    cov_u: np.ndarray = None,
    cov_overrides_u: list = [],
):
    """Generate datasets with known m and u probabilities to feed into the Fellegi Sunter model
    Uses a Splink settings objects to configure data generation

    Args:
        num_rows (int): Number of rows to generate
        settings (dict): Splink settings dictionary
        cov_m (np.ndarray, optional): A covariance matrix for matches, shape of n x n where n is number of comparison columns.  Defaults to None.
        cov_overrides_m (list, optional): Specific cov_overrides for matches to apply to the default covariance identity matrix. e.g. [["first_name", "surname", 0.5], ["first_name", "dob", 0.1]]. Defaults to [].
        cov_u (np.ndarray, optional): A covariance matrix for non matches, shape of n x n where n is number of comparison columns.  Defaults to None.
        cov_overrides_u (list, optional): Specific cov_overrides for non matches to apply to the default covariance identity matrix. e.g. [["first_name", "surname", 0.5], ["first_name", "dob", 0.1]]. Defaults to [].

    Returns:
        pd.DataFrame: A pandas dataframe representing df_gammas
    """

    settings = _add_essentials_to_settings(settings)
    settings = complete_settings_dict(settings, None)

    num_matches = math.floor(num_rows * settings["proportion_of_matches"])
    num_non_matches = math.ceil(num_rows * (1 - settings["proportion_of_matches"]))

    if not cov_m:
        cov_m = _gen_cov_matrix_no_correlation(settings)

    if not cov_u:
        cov_u = _gen_cov_matrix_no_correlation(settings)

    for override in cov_overrides_m:
        _apply_override(override, cov_m, settings)

    for override in cov_overrides_u:
        _apply_override(override, cov_u, settings)

    ran_data = _generate_uniform_random_numbers(num_matches, cov_m)
    matches = _ran_numbers_to_gammas(ran_data, settings, True)
    df_m = pd.DataFrame(matches)
    df_m["true_match_l"] = 1
    df_m["true_match_r"] = 1

    ran_data = _generate_uniform_random_numbers(num_non_matches, cov_u)
    non_matches = _ran_numbers_to_gammas(ran_data, settings, False)
    df_nm = pd.DataFrame(non_matches)
    df_nm["true_match_l"] = 0
    df_nm["true_match_r"] = 0

    df_all = pd.concat([df_m, df_nm])

    df_all["unique_id_l"] = [str(uuid4())[:8] for _ in range(len(df_all.index))]
    df_all["unique_id_r"] = [str(uuid4())[:8] for _ in range(len(df_all.index))]

    return df_all
