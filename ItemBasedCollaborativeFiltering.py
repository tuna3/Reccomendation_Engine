import sys
from pyspark import SparkContext
import os
from itertools import groupby, combinations
from operator import add
import time
import numpy as np
import random




class ItemBasedCF:
    def __init__(self, inputFile, testFile):
        self.inputFile = inputFile
        self.testFile =  testFile
        self.correlation ={}
        self.numHash = 20
        self.numBand = 5
        self.hashFunction =[]

    def generate_hash_functions(self):
        i = 0
        while i != self.numHash:
            a = random.randint(1,self.totalUsers-10)
            b = random.randint(1,self.totalUsers-10)
            func = (a,b)
            if func not in self.hashFunction:
                self.hashFunction.append(func)
                i += 1


    def create_signature(self,inputData):
        #input = users per movie
        SignaturePerMovie =[]
        movieEntry = np.array(inputData)
        for n in range(self.numHash):
            hash = (self.hashFunction[n][0]*movieEntry + self.hashFunction[n][1])%self.totalUsers
            SignaturePerMovie.append(np.min(hash))
        inputData = SignaturePerMovie
        return SignaturePerMovie

    def hashBucket(self,inputData,a,b):
        entry = np.array(inputData)
        hash = ((a * entry) + b) % 3000
        return np.min(hash)

    def computeJacardSimilarity(self,movie1,movie2):
        Users1 = set(self.movieDict[movie1])
        Users2 = set(self.movieDict[movie2])
        return float(len(Users1 & Users2))/float(len(Users1 | Users2))

    def ComputeCorellation(self,Movie1, Movie2):
        dictMovie1 = dict(Movie1)
        dictMovie2 = dict(Movie2)
        coratedItem = set(dictMovie1.keys()) & set(dictMovie2.keys())
        flagContain = False


        if len(coratedItem) == 0 :
            return 0
        Valu = []
        Valv = []
        for user in coratedItem :
            Valu.append(dictMovie1[user])
            Valv.append(dictMovie2[user])
        Valu = np.array(Valu)
        Valv = np.array(Valv)

        meanU = np.mean(Valu)
        meanV = np.mean(Valv)

        Valu = Valu - meanU
        Valv = Valv - meanV

        numerator = np.sum(np.multiply(Valu,Valv))
        denomU = np.sqrt(np.sum(Valu**2))
        denomV = np.sqrt(np.sum(Valv**2))

        if denomU * denomV == 0:
            return 0
        return float(numerator)/float(denomU*denomV)

    def ComputeAllCorrelations(self):
        Movies = sorted(list(self.MovieDict.keys()))
        MoviePair = list(combinations(Movies,2))

        for pair in MoviePair:
            corr = self.ComputeCorellation(self.MovieDict[pair[0]],self.MovieDict[pair[1]])
            self.correlation[frozenset(pair)] = corr




    def predictValue(self,UserId,movie):
        UserRatedItems = self.UserDict[UserId]
        UserRatedItemsDict = dict(UserRatedItems)
        correlationforMovie =[]
        numerator = 0
        denominator = 0

        movieUsers = self.MovieDict[movie]

        for key in UserRatedItemsDict.keys():
            corr = self.ComputeCorellation(self.MovieDict[key], movieUsers)
            val = UserRatedItemsDict[key]
            numerator += (corr*val)
            denominator += np.abs(corr)

        if denominator == 0:
            UserRating = np.mean(np.array(self.UserDict[UserId]),0)[1]
            return UserRating
        return float(numerator)/float(denominator)



    def JpredictValue(self,UserId,movie):
        UserRatedItems = self.UserDict[UserId]
        UserRatedItemsDict = dict(UserRatedItems)
        correlationforMovie =[]
        numerator = 0
        denominator = 0

        for key in UserRatedItemsDict.keys():
            if frozenset([key,movie]) in self.JSimilDict.keys():
                #print('present matched')
                corr = self.JSimilDict[frozenset([key,movie])]
                val = UserRatedItemsDict[key]
                numerator += (corr*val)
                denominator += np.abs(corr)

        if denominator == 0:
            UserRating = np.mean(np.array(self.UserDict[UserId]),0)[1]
            return UserRating
        return float(numerator)/float(denominator)


    def run(self):
        sc = SparkContext("local[8]", "ratings")
        sc._conf.set('spark.executor.memory','5g').set('spark.driver.memory','4g').set('spark.driver.maxResultsSize','0')
        inputFile = sys.argv[1]
        testFile = sys.argv[2]

        inputRDD = sc.textFile(inputFile)
        testRDD = sc.textFile(testFile)

        header = inputRDD.first() #extract header
        inputRDD = inputRDD.filter(lambda row : row != header)

        header2 = testRDD.first() #extract header
        testRDD = testRDD.filter(lambda row : row != header2)


        inputRDD = inputRDD.map(lambda line: line.split(',')).map(lambda x:  ((int(x[0]), int(x[1])),float(x[2])) )
        testRDD = testRDD.map(lambda line: line.split(',')).map(lambda x:  ((int(x[0]),int(x[1])),1))

        input = inputRDD.subtractByKey(testRDD)


        MovieInput = input.map(lambda x: (x[0][1],(x[0][0], x[1]))).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
        self.MovieDict = dict(MovieInput.collect())





        UserInput = input.map(lambda x: (x[0][0],(x[0][1], x[1]))).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
        self.UserDict = dict(UserInput.collect())

        # Jaccard Similarity
        mapping = inputRDD.map(lambda x:  (x[0][0],x[0][1])).groupByKey()
        movieInput = inputRDD.map(lambda x:  (x[0][1],x[0][0])).groupByKey().map(lambda x: (x[0],list(x[1]))).sortByKey()
        self.totalUsers = len(mapping.collect())
        print(self.totalUsers)
        self.generate_hash_functions()


        Signature = movieInput.map(lambda x : (x[0],self.create_signature(x[1])))

        bandSize = self.numHash//self.numBand
        unique = sc.emptyRDD()
        self.movieDict = dict(movieInput.collect())
        start = time.time()
        for i in range(self.numBand):
            bands = Signature.map(lambda x : (x[0], x[1][(i * bandSize) : ((i+1) * bandSize) ]))
            a = random.randint(1,1500)
            b = random.randint(1,1000)
            bands = bands.map(lambda x:(x[0], self.hashBucket(x[1],a,b))).map(lambda x: (x[1],x[0])).groupByKey().map(lambda x:sorted(list(x[1]))).flatMap(lambda x: list(combinations(x,2)))
            unique = unique.union(bands).distinct()
        unique = unique.distinct().map(lambda x :((x[0],x[1]) ,self.computeJacardSimilarity(x[0],x[1]))).filter(lambda x : x[1] >= 0.5)
        output = unique.map(lambda x: (frozenset([x[0][0], x[0][1]]),x[1])).collect()
        self.JSimilDict = dict(output)

        print('Jacard Similarity computed')

        ###
        #self.JSimilDict = l.run(inputRDD)
        start = time.time()
        testRDD1 = testRDD.map(lambda x: ((x[0][0], x[0][1]),self.JpredictValue(x[0][0], x[0][1])))

        WritingOutput = testRDD1.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().map(lambda x : (x[0], sorted(list(x[1]),key=lambda tup: tup[0]))).sortByKey().collect()

        with open('Tuhina_Kumar_ItemBasedCF.txt','w') as f:
            for i in range(len(WritingOutput)):
                str1 =''
                for j in range(len(WritingOutput[i][1])):
                    str1 = str1 + str(WritingOutput[i][0]) +', ' + str(WritingOutput[i][1][j][0])+', '+ str(WritingOutput[i][1][j][1])+'\n'
                f.write(str1)

        print(" computing jaccard",time.time() - start)
        error1 = testRDD1.join(inputRDD)
        print(" with join", time.time() - start)
        print(len(error1.collect()))
        print('Jacard Value computed')

        MSE = error1.map(lambda r: (r[1][0] - r[1][1])**2).mean()

        RMSE = np.sqrt(MSE)
        print(' Jacard RMSE',RMSE)

        diff = error1.map(lambda r: ( np.clip(np.floor(np.abs(r[1][0] - r[1][1])),0,4), 1)).reduceByKey(lambda a,b : a+b).sortByKey().collect()

        for i in range(len(diff)):
            print(diff[i])


        #pearson correlation
        #self.ComputeAllCorrelations()
        start = time.time()
        testRDD2 = testRDD.map(lambda x: ((x[0][0], x[0][1]), self.predictValue(x[0][0],x[0][1])))
        print(" computing value pearson,", time.time() - start)
        error2 = testRDD2.join(inputRDD)
        print('Pearson error computed', len(error2.collect()), time.time()- start)

        MSE2 = error2.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        RMSE2 = np.sqrt(MSE2)
        print(" Pearson RMSE" , RMSE2)

        diff2= error2.map(lambda r: ( np.clip(np.floor(np.abs(r[1][0] - r[1][1])),0,4), 1)).reduceByKey(lambda a,b : a+b).sortByKey().collect()

        for i in range(len(diff2)):
            print(diff2[i])




if __name__ == '__main__' :
    start = time.time()
    inputFile = sys.argv[1]
    testFile = sys.argv[2]

    CF_Jaccard = ItemBasedCF(inputFile, testFile)
    CF_Jaccard.run()

    print(time.time() - start)
