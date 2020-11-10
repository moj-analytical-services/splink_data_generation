from splink import Splink
from pandas import DataFrame

from pyspark.context import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql import Row

from splink.iterate import iterate

import logging

logging.basicConfig()  # Means logs will print in Jupyter Lab

# Set to DEBUG if you want splink to log the SQL statements it's executing under the hood
logging.getLogger("splink").setLevel(logging.INFO)
from splink import Splink


def estimate(df_gammas: DataFrame, settings: dict):
    """Take pandas datafrae of gammas and estimate splink model

    Args:
        df_gammas (DataFrame): df_gammas
        settings (dict): Splink settings dictionary
    """

    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

    df = spark.createDataFrame(df_gammas)

    linker = Splink(settings, spark=spark, df=df)

    df_e = iterate(df, linker.params, linker.settings, spark)

    return df_e, linker
