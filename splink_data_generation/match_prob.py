from copy import deepcopy


def add_match_prob(df, settings):
    settings = deepcopy(settings)
    lam = df["true_match_l"].mean()
    settings["proportion_of_matches"] = lam

    # Formula is λ*m1*m2/(λ*m1*m2 + (1-λ)(u1*u2)), see https://imai.fas.harvard.edu/research/files/linkage.pdf
    # If you divide through by u1*u2, and call m1/u1 = bf1
    # then it's λ*b1*b2/(λ*b1*b2 + (1-λ)
    # But can be calculated iteratively in same way as naive bayes classifier
    # Note that  λ*b1/(λ*b1 + (1-λ) = a
    # a*b2/(a*b2) + (1-a) = λ*b1*b2/(λ*b1*b2 + (1-λ)
    # So that you can calculate iteratively:
    # old_p = λ
    # Update with b = m1/u1
    # new_p = old_p * b / (old_p * b) + (1-old_p)
    # For further info see https://observablehq.com/@robinl/conditional-independence-and-repeated-application-of-bay

    # Look up relevant probabilities using dict
    df["_p"] = lam
    for cc in settings["comparison_columns"]:
        m_lookup = {i: v for i, v in enumerate(cc["m_probabilities"])}
        u_lookup = {i: v for i, v in enumerate(cc["u_probabilities"])}
        if "custom_name" in cc:
            name = cc["custom_name"]
        else:
            name = cc["col_name"]
        df["_m"] = df[f"gamma_{name}"].map(m_lookup)
        df["_u"] = df[f"gamma_{name}"].map(u_lookup)
        df["_b"] = df["_m"] / df["_u"]

        df["_p"] = (df["_p"] * df["_b"]) / ((df["_p"] * df["_b"]) + (1 - df["_p"]))

    df["true_match_probability_l"] = df["_p"]
    df["true_match_probability_r"] = df["_p"]

    df = df.drop(["_p", "_m", "_u", "_b"], axis=1)

    return df