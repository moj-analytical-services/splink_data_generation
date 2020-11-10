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


def generate_df_gammas(
    num_rows: int, settings: dict, cov: np.ndarray = None, corr_overrides: list = []
):
    """Generate datasets with known m and u probabilities to feed into the Fellegi Sunter model
    Uses a Splink settings objects to configure data generation

    Args:
        num_rows (int): Number of rows to generate
        settings (dict): Splink settings dictionary
        cov (np.ndarray, optional): A covariance matrix, shape of n x n where n is number of comparison columns.  Defaults to None.
        corr_overrides (list, optional): Specific corr_overrides to apply to the default covariance matrix which is an identity matrix. e.g. [["first_name", "surname", 0.5], ["first_name", "dob", 0.1]]. Defaults to [].

    Returns:
        pd.DataFrame: A pandas dataframe representing df_gammas
    """

    settings = _add_essentials_to_settings(settings)
    settings = complete_settings_dict(settings, None)

    num_matches = math.floor(num_rows * settings["proportion_of_matches"])
    num_non_matches = math.ceil(num_rows * (1 - settings["proportion_of_matches"]))

    if not cov:
        cov = _gen_cov_matrix_no_correlation(settings)

    for ov in corr_overrides:
        _apply_override(ov, cov, settings)

    ran_data = _generate_uniform_random_numbers(num_matches, cov)
    matches = _ran_numbers_to_gammas(ran_data, settings, True)
    df_m = pd.DataFrame(matches)
    df_m["true_match_l"] = 1
    df_m["true_match_r"] = 1

    ran_data = _generate_uniform_random_numbers(num_non_matches, cov)
    non_matches = _ran_numbers_to_gammas(ran_data, settings, False)
    df_nm = pd.DataFrame(non_matches)
    df_nm["true_match_l"] = 0
    df_nm["true_match_r"] = 0

    df_all = pd.concat([df_m, df_nm])

    df_all["unique_id_l"] = [str(uuid4())[:8] for _ in range(len(df_all.index))]
    df_all["unique_id_r"] = [str(uuid4())[:8] for _ in range(len(df_all.index))]

    return df_all
