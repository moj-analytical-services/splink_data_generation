from copy import deepcopy
import numpy as np


def add_log_likelihood(df, settings):
    settings = deepcopy(settings)
    lam = df["true_match_l"].mean()
    settings["proportion_of_matches"] = lam

    # Formula from https://imai.fas.harvard.edu/research/files/linkage.pdf
    # For further info see https://observablehq.com/@robinl/conditional-independence-and-repeated-application-of-bay

    df["_pm"] = lam
    df["_pu"] = 1 - lam
    for cc in settings["comparison_columns"]:
        m_lookup = {i: v for i, v in enumerate(cc["m_probabilities"])}
        u_lookup = {i: v for i, v in enumerate(cc["u_probabilities"])}
        if "custom_name" in cc:
            name = cc["custom_name"]
        else:
            name = cc["col_name"]
        df["_m"] = df[f"gamma_{name}"].map(m_lookup)
        df["_u"] = df[f"gamma_{name}"].map(u_lookup)
        df["_pm"] = df["_m"] * df["_pm"]
        df["_pu"] = df["_u"] * df["_pu"]

    df["true_log_likelihood_l"] = np.log(df["_pm"] + df["_pu"])
    df["true_log_likelihood_r"] = df["true_log_likelihood_l"]

    df = df.drop(["_m", "_u", "_pm", "_pu"], axis=1)

    return df