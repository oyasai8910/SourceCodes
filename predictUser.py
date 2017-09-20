#!/usr/bin/env python

from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql import DataFrameWriter
import pandas as pd
import math


path_of_target_jans = "cola_jans.csv"
user_number_per_product = 2000
postfix = "cola_2k"

if __name__ == "__main__":
    sc = SparkContext(appName="PythonCollabrativeFilteringPredictUsers")
    model = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")


    jan_ = pd.read_csv("jan.csv")
    id_ = pd.read_csv("id.csv")


    target_jans = pd.read_csv(path_of_target_jans, header = None)
    target_jans = target_jans.merge(jan_, left_on=0, right_on="JAN", how="inner")
    target_jans.to_csv("conf.csv")


    df = pd.DataFrame()
    eliminate = []
    for jan in target_jans["JAN (to number)"]:
        if jan in eliminate:  # I don't know why model.recommendUsers doesn't work in the case of 1065 only.
            continue
        recommend_users = model.recommendUsers(int(jan), user_number_per_product)
        df_add = pd.DataFrame(recommend_users)

        print(jan, "was excuted")

        df = pd.concat([df, df_add])


    df = df.merge(jan_, left_on="product", right_on="JAN (to number)", how="inner")
    df = df.merge(id_, left_on="user", right_on="会員ID (to number)", how="inner")


    df_uniqueid = df["会員ID"].drop_duplicates()

    df.to_csv('predictUsers_' + postfix + '.csv')

    df_sort = df.sort_values(by='rating', ascending=False)
    df_sort = df_sort.drop_duplicates(subset=['会員ID'])
    df_sort.to_csv('sortPredictUsers_' + postfix + '.csv')

    df_uniqueid.to_csv('uniqueid_' + postfix + '.csv')

