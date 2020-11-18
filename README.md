# splink_data_generation


[`splink_data_generation`](https://github.com/moj-analytical-services/splink_data_generation) is a Python package that is able to generate datasets with known parameters.

It has two main ways of generating data:
- `generate_df_gammas_exact`, which produces a dataset with parameters that precisely match the `m` and `u` values specified, and that precisely obeys the assumption of independence of comparison of linking variables conditional on match status.  This function has three limitations:

    (1) It's not possible to control the overall proportion of matches in the output dataset, and

    (2) It's not possible to generate data that breaks the assumption of conditional independence.

    (3) As the complexity of the dataset requested increases, the number of rows generated to satisfy the assumptions can be very high

- `generate_df_gammas_random`, which produces rows at random using the data generating mechanism specified by the supplied parameters.  This function allows the user to specify the proportion of matches, and a covariance matrix which dictates correlations between linking variables conditional on match status.  Since rows are generated at random, the parameters of the resultant dataset will converge to true parameters and the number of rows generated tends to infinity.

