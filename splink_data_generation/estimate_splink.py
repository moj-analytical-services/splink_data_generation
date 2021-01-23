from splink import Splink
from pandas import DataFrame

from pyspark.sql import SparkSession

from splink.iterate import iterate
from splink.model import Model


def estimate(
    df_gammas: DataFrame, settings: dict, spark: SparkSession, compute_ll=True
):
    """Take pandas datafrae of gammas and estimate splink model

    Args:
        df_gammas (DataFrame): Pandas dataframe of df_gammas
        settings (dict): Splink settings dictionary
        spark (SparkSession): SparkSession object
    """

    settings["retain_matching_columns"] = False

    df = spark.createDataFrame(df_gammas)

    model = Model(settings, spark)

    df_e = iterate(df, model, spark, compute_ll=compute_ll)

    return df_e, model
