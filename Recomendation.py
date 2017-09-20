#!/usr/bin/env python

from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RegressionMetrics
import math

TRAIN = 8
VALIDATION = 0
TEST = 2
ADJUST = 10 - TRAIN - VALIDATION - TEST
SPLIT_LIST = TRAIN, VALIDATION, TEST, ADJUST

if __name__ == "__main__":
    sc = SparkContext(appName="PythonCollaborativeFilteringExample")
    # $example on$
    # Load and parse the data
    data = sc.textFile("data.csv")

    
    # Splitting the data
    ratings = data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    train, validation, test, adjustment = ratings.randomSplit(SPLIT_LIST) # splitting data into testing data, validation data and training data

    train.cache() # caching data for quick optimization
    validation.cache()
    test.cache()
    
    validationForPredict = validation.map(lambda x: (x[0], x[1]))
    actualReformatted = validation.map(lambda x: ((x[0], x[1]), x[2]))

    # Build the recommendation model using Alternating Least Squares
    #rank = 10
    #numIterations = 10
    #model = ALS.train(train, rank, numIterations)

    iterations = [5, 7, 10]
    regularizationParameter = 0.05
    ranks = [10, 12, 15]
    RMSEs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    err = 0
    tolerance = 0.03

    minRMSE = float('inf')
    bestIteration = -1
    bestRank = -1
    ptr1 = "output \n"

    #validating hyper-parameters
    #for rank in ranks:
    #    for iteration in iterations:
    #        model = ALS.trainImplicit(train, rank, iteration, lambda_=regularizationParameter)
    #        predictedRatings = model.predictAll(validationForPredict)
    #        predictedReformatted = predictedRatings.map(lambda x: ((x[0], x[1]), x[2]))

    #        predictionAndObservations = (predictedReformatted.join(actualReformatted).map(lambda x: x[1]))
    #        
    #        metrics = RegressionMetrics(predictionAndObservations)
    #        RMSE = metrics.rootMeanSquaredError
    #        RMSEs[err] = RMSE
    #        err += 1
    #        
    #        #print ("For rank %s and iteration %s, the RMSE is %s") % (rank, iteration, RMSE)
    #        ptr1 = ptr1 +  "For rank " + str(rank) +  " and iterations " + str(iteration) + " the RMSE is " + str(RMSE) + " \n"
    #        print(ptr1)
    #        if RMSE < minRMSE:
    #            minRMSE = RMSE
    #            bestIteration = iteration
    #            bestRank = rank

    ###print ("The best model was trained with rank %s and iteration %s") % (bestRank, bestIteration)
    #ptr2 = "The best model was trained with rank " + str(bestRank) + " and iteration " + str(bestIteration) + " \n"
    #print(ptr2)

    bestRank = 25
    bestIteration = 15
    bestModel = ALS.trainImplicit(train, bestRank, iterations=bestIteration, lambda_=regularizationParameter)

    testForPredicting = test.map(lambda x: (x[0], x[1]))
    testReformatted = test.map(lambda x: ((x[0], x[1]), x[2]))

    predictedTest = bestModel.predictAll(testForPredicting)
    
    predictedTestReformatted = predictedTest.map(lambda x: ((x[0], x[1]), x[2]))

    predictionAndObservationTest = (predictedTestReformatted.join(testReformatted). map(lambda x: x[1]))

    metrics = RegressionMetrics(predictionAndObservationTest)
    testRMSE = metrics.rootMeanSquaredError

    print ("The Model had a RMSE on the test set of " + str(testRMSE))
    ptr3 = "The Model had a RMSE on the test set of " + str(testRMSE)
    with open("RMSE_vs_train" + ".csv", "a") as f:
        f.write(','.join(map(str, SPLIT_LIST + (testRMSE,))) + '\n')

    # Evaluate the model on training data
    #testdata = test.map(lambda p: (p[0], p[1]))
    #predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    
    # Keep the it if there is already a data, if not put in the data camputed by ALS
    #ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

    #RMSE = math.sqrt(ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    #print("Root Mean Squared Error = " + str(RMSE))

    # Save and load model
    bestModel.save(sc, "target/tmp/myCollaborativeFilter")
    #sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter") 
    # $example off$

