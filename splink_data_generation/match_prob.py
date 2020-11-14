from copy import deepcopy


def _row_match_probability(r, settings):

    λ = settings["proportion_of_matches"]
    cc = settings["comparison_columns"]

    cc = {c["col_name"]: c for c in cc}

    gamma_cols = [k for k in r.keys() if "gamma_" in k]

    numerator = λ
    for col in gamma_cols:
        col_name = col.replace("gamma_", "")
        val = r[col]
        p = cc[col_name]["m_probabilities"][val]
        numerator = numerator * p

    denominator = 1 - λ
    for col in gamma_cols:
        col_name = col.replace("gamma_", "")
        val = r[col]
        p = cc[col_name]["u_probabilities"][val]
        denominator = denominator * p

    r["true_match_probability_l"] = numerator / (numerator + denominator)
    r["true_match_probability_r"] = numerator / (numerator + denominator)

    return r


def add_match_prob(df, settings):
    settings = deepcopy(settings)
    settings["proportion_of_matches"] = df["true_match_l"].mean()
    print(settings)
    return df.apply(_row_match_probability, axis=1, settings=settings)