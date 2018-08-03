import sys
from pyspark import SparkContext
import os
from itertools import groupby, combinations
from operator import add
import time
import numpy as np
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


if __name__ == '__main__' :

    inputFile = sys.argv[1]
    testFile = sys.argv[2]

    #conf = SparkConf().setMaster("local[8]").setAppName("ratings").set("spark.executor.memory", "6g")

    sc = SparkContext("local[8]", "ratings")
    sc._conf.set('spark.executor.memory','5g').set('spark.driver.memory','4g').set('spark.driver.maxResultsSize','0')

    inputRDD = sc.textFile(inputFile)
    testRDD = sc.textFile(testFile)

    header = inputRDD.first() #extract header
    inputRDD = inputRDD.filter(lambda row : row != header)

    header2 = testRDD.first() #extract header
    testRDD = testRDD.filter(lambda row : row != header2)

    inputRDD = inputRDD.map(lambda line: line.split(',')).map(lambda x:  ((int(x[0]), int(x[1])),float(x[2])) )
    testRDD = testRDD.map(lambda line: line.split(',')).map(lambda x:  ((int(x[0]),int(x[1])),1))

    input1 = inputRDD.subtractByKey(testRDD)
    input = input1.map(lambda x: Rating(x[0][0], x[0][1], x[1]))
    sc.setCheckpointDir('/tmp')

    rank = 8
    numIterations = 10
    lmbda = 0.1
    numBlocks = 16
    nonnegative = True
    model = ALS.train(input, rank, numIterations, lmbda,nonnegative = True, seed = 42)

    testRDD = testRDD.map(lambda x: (x[0][0], x[0][1])).distinct()
    predictions = model.predictAll(testRDD).map(lambda x: ( (x[0], x[1]), x[2])).mapValues(lambda v: np.clip(v,0,5))

    '''
    pred2 = testRDD.map(lambda x :((x[0],x[1]),1)).subtractByKey(predictions)

    UserInput = input1.map(lambda x: (x[0][0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
    UserInput = dict(UserInput.collect())

    UserAvg ={}
    for k in UserInput.keys():
        UserMovie = np.mean(np.array(UserInput[k]))
        UserAvg[k] = UserMovie



    pred2 = pred2.map(lambda x: ((x[0][0],x[0][1]),UserAvg[x[0][0]]))
    predictions = predictions.union(pred2).sortByKey()



    print(" pred",len(predictions.collect()), " pred 2", len(pred2.collect()),pred2.take(10))
    '''

    WritingOutput = predictions.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().map(lambda x : (x[0], sorted(list(x[1]),key=lambda tup: tup[0]))).sortByKey().collect()

    with open('Tuhina_Kumar_ModelBasedCF_Big.txt','w') as f:
        for i in range(len(WritingOutput)):
            str1 =''
            for j in range(len(WritingOutput[i][1])):
                str1 = str1 + str(WritingOutput[i][0]) +', ' + str(WritingOutput[i][1][j][0])+', '+ str(WritingOutput[i][1][j][1])+'\n'
            f.write(str1)


    error = predictions.join(inputRDD)
    print(len(error.collect()))

    MSE = error.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    RMSE = np.sqrt(MSE)
    print('RMSE', RMSE)

    diff = error.map(lambda r: ( np.clip(np.floor(np.abs(r[1][0] - r[1][1])),0,4), 1)).reduceByKey(lambda a,b : a+b).sortByKey().collect()

    for i in range(len(diff)):
        print(diff[i])
    print('>=0 and <1:',diff[0][1])
    print('>=1 and <2:', diff[1][1])
    print('>=2 and <3:', diff[2][1])
    print('>=3 and <4:', diff[3][1])
    print('>=4', diff[4][1])
