# Generate a dataset with given m and u probabilities
# that conforms exactly with the assumptions of the
# Fellegi-Sunter model (i.e. conditional independence
# of comparison vector values given match status
from uuid import uuid4
from fractions import Fraction
import numpy as np
import pandas as pd
from functools import reduce
import warnings
from copy import deepcopy


def _validate_sum(settings, limit_denominator):
    cc = settings["comparison_columns"]

    for c in cc:
        for ptype in ["m_probabilities", "u_probabilities"]:
            probs = c[ptype]
            probs = [
                abs(float(Fraction(p).limit_denominator(limit_denominator)))
                for p in probs
            ]
            err = sum(probs) - 1.0
            if abs(err) > 0.0001:
                col_name = c["col_name"]
                raise ValueError(
                    (
                        f"The sum of probabilities for {col_name} {ptype} "
                        f"is {sum(probs):,.6f}.  It should sum to 1.0."
                    )
                )


def _num_rows(vectors):
    rowcount_m = reduce(lambda x, y: x * len(y), vectors["m_vectors"], 1)
    rowcount_u = reduce(lambda x, y: x * len(y), vectors["u_vectors"], 1)
    return rowcount_m + rowcount_u


def _validate_probs(probs, limit_denominator):
    for p in probs:
        err = abs(float(Fraction(p).limit_denominator(limit_denominator)) - p)
        if err > 0.000001:
            warnings.warn(
                (
                    f"Probability {p} converted into "
                    f"fraction {Fraction(p).limit_denominator(limit_denominator)} with error {err}"
                )
            )


def _generate_vector(probs, limit_denominator):
    _validate_probs(probs, limit_denominator)
    f_list = [Fraction(f).limit_denominator(limit_denominator) for f in probs]
    d_list = [fr.denominator for fr in f_list]
    num_values_to_generate = np.lcm.reduce(d_list)

    num_values = [int(f * num_values_to_generate) for f in f_list]

    arr = []
    for i, val in enumerate(num_values):
        arr.extend([i] * val)

    return arr


def _generate_vectors(settings, limit_denominator):

    cc = settings["comparison_columns"]

    results = {"m_vectors": [], "u_vectors": []}
    for c in cc:
        v = _generate_vector(c["m_probabilities"], limit_denominator)
        results["m_vectors"].append(v)

        v = _generate_vector(c["u_probabilities"], limit_denominator)
        results["u_vectors"].append(v)

    return results


def generate_df_gammas_exact(settings, max_rows=1e6, limit_denominator=100):

    # do not modify settings object
    settings = deepcopy(settings)

    # Value is irrelvant, it won't be used.  Needed for settings to validate
    settings["proportion_of_matches"] = 0.0

    _validate_sum(settings, limit_denominator)
    vectors = _generate_vectors(settings, limit_denominator)

    num_rows = _num_rows(vectors)
    if num_rows > max_rows:
        raise ValueError(
            (
                "The m and u probabilities specified will generate "
                f"{num_rows:,.0f} rows, which is greater than the max_rows "
                f"value of {max_rows:,.0f}.  If you really want to generate "
                "this many rows, increase the max_rows parameter "
                "of generate_df_gammas_exact"
            )
        )

    cc = settings["comparison_columns"]

    num_cols = len(cc)
    col_names = [c["col_name"] for c in cc]

    col_names = [f"gamma_{c}" for c in col_names]
    mg_m = np.meshgrid(*vectors["m_vectors"])
    mg_m = np.array(mg_m).T.reshape(-1, num_cols)
    df_m = pd.DataFrame(mg_m, columns=col_names)
    df_m["true_match_l"] = 1
    df_m["true_match_r"] = 1

    mg_u = np.meshgrid(*vectors["u_vectors"])
    mg_u = np.array(mg_u).T.reshape(-1, num_cols)
    df_u = pd.DataFrame(mg_u, columns=col_names)
    df_u["true_match_l"] = 0
    df_u["true_match_r"] = 0

    df_all = pd.concat([df_m, df_u])
    df_all["unique_id_l"] = [str(uuid4())[:8] for _ in range(len(df_all.index))]
    df_all["unique_id_r"] = [str(uuid4())[:8] for _ in range(len(df_all.index))]

    match_prop = df_all["true_match_l"].mean()

    warnings.warn(
        f"Note that the proportion_of_matches setting is ignored by this generator. "
        "Only the m_probabilities and u_probabilities are observed. "
        f"The proportion of matches in the generated dataset was {match_prop:,.3f}"
    )

    return df_all
