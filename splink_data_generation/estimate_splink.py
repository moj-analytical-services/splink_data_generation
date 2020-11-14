from splink import Splink
from pandas import DataFrame

from pyspark.context import SparkContext
from pyspark.sql import SparkSession

from splink.iterate import iterate

from splink import Splink


def estimate(df_gammas: DataFrame, settings: dict, spark: SparkSession):
    """Take pandas datafrae of gammas and estimate splink model

    Args:
        df_gammas (DataFrame): df_gammas
        settings (dict): Splink settings dictionary
        spark (SparkSession): SparkSession object
    """

    df = spark.createDataFrame(df_gammas)

    linker = Splink(settings, spark=spark, df=df)

    df_e = iterate(df, linker.params, linker.settings, spark)

    return df_e, linker