from copy import deepcopy
import math


def _row_ll(r, settings):

    λ = settings["proportion_of_matches"]
    cc = settings["comparison_columns"]

    cc = {c["col_name"]: c for c in cc}

    gamma_cols = [k for k in r.keys() if "gamma_" in k]

    # Prob of match
    prob_match = λ
    for col in gamma_cols:
        col_name = col.replace("gamma_", "")
        val = r[col]
        p = cc[col_name]["m_probabilities"][val]
        prob_match = prob_match * p

    prob_non_match = 1 - λ
    for col in gamma_cols:
        col_name = col.replace("gamma_", "")
        val = r[col]
        p = cc[col_name]["u_probabilities"][val]
        prob_non_match = prob_non_match * p

    r["true_log_likelihood_l"] = math.log(prob_match + prob_non_match)
    r["true_log_likelihood_r"] = math.log(prob_match + prob_non_match)

    return r


def add_log_likelihood(df, settings, set_prop_from_true=True):
    settings = deepcopy(settings)
    if set_prop_from_true:
        settings["proportion_of_matches"] = df["true_match_l"].mean()
    return df.apply(_row_ll, axis=1, settings=settings)