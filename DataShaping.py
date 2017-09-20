#!/usr/bin/env python

from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import math

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName('create_cf_model')\
        .config('spark.sql.crossJoin.enabled', 'true')\
        .config('spark.debug.maxToStringFields', 500)\
        .getOrCreate()
    sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

    df = sqlContext\
        .read\
        .format('com.databricks.spark.csv')\
        .options(header="true")\
        .load("allstores_201707.csv")

    df = df.select("日", "JAN", "会員ID")
    df_renamed = df.withColumnRenamed("日", "date")\
        .withColumnRenamed("会員ID", "id")\
        .withColumnRenamed("JAN", "jan")

    df_grouped = df_renamed.groupBy("id", "jan").count()

    df_grouped.cache()
    
    unique_users_rdd = df_grouped.rdd.map(lambda l: l[0]).distinct().zipWithIndex() 
    unique_users_df = sqlContext.createDataFrame(
        unique_users_rdd,
        ('user', 'unique_user_id')
    )
    
    unique_jans_rdd = df_grouped.rdd.map(lambda l: l[1]).distinct().zipWithIndex() 
    unique_jans_df = sqlContext.createDataFrame(
        unique_jans_rdd,
        ('jan', 'unique_jan_id') 
    )

    df_grouped = df_grouped.join(
        unique_users_df,
        df_grouped["id"] == unique_users_df["user"],
        'inner'
    ).drop(unique_users_df['user'])
 
    df_grouped = df_grouped.join(
        unique_jans_df,
        df_grouped["jan"] == unique_jans_df["jan"],
        'inner'
    ).drop(unique_jans_df['jan'])

    # Data to CSV
    df_grouped.repartition(1)\
        .select("unique_user_id", "unique_jan_id", "count")\
        .write\
        .format("com.databricks.spark.csv")\
        .options(header="true")\
        .save("data_spark")

    # Unique User ID
    unique_users_df.repartition(1)\
        .write\
        .format("com.databricks.spark.csv")\
        .options(header="true")\
        .save("user")

    # Unique JAN ID
    unique_jans_df.repartition(1)\
        .write\
        .format("com.databricks.spark.csv")\
        .options(header="true")\
        .save("jan")
