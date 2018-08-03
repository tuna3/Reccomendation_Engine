import sys
from pyspark import SparkContext
import os
from itertools import groupby, combinations
from operator import add
import time
import numpy as np

class UserBasedCF:
    def __init__(self, inputFile, testFile):
        self.inputFile = inputFile
        self.testFile =  testFile
        self.correlation ={}

    def ComputeCorellation(self,User1, User2):
        dictUser1 = dict(User1)
        dictUser2 = dict(User2)

        coratedItem = set(dictUser1.keys()) & set(dictUser2.keys())
        flagContain = False


        if len(coratedItem) == 0 :
            return 0
        Valu = []
        Valv = []
        for movie in coratedItem :
            Valu.append(dictUser1[movie])
            Valv.append(dictUser2[movie])
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
        Users = sorted(list(self.UserDict.keys()))
        UserPair = list(combinations(Users,2))

        for pair in UserPair:
            corr = self.ComputeCorellation(self.UserDict[pair[0]],self.UserDict[pair[1]])
            self.correlation[frozenset(pair)] = corr




    def predictValue(self, UserId, movie):
        UserMovie = self.UserDict[UserId]
        correlationForUser =[]
        for key in self.UserDict.keys():
            if key != UserId:
                movierating = dict(self.UserDict[key])
                if movie in movierating.keys():
                    corr = self.correlation[frozenset([key, UserId])]
                    val = movierating[movie]
                    correlationForUser.append((key,corr, val))
        #print(" correlation computed")
        correlationForUser = sorted(correlationForUser, key=lambda tup: tup[1])

        k = min(50,len(correlationForUser))
        denom = 0
        num = 0
        for i in range(k):
            val = self.UserDict[correlationForUser[i][0]]
            val = np.array(val)
            meanval = float(np.sum(val,0)[1] - correlationForUser[i][2])/float(len(val))

            num += (correlationForUser[i][2] -meanval)* correlationForUser[i][1]
            denom += np.abs(correlationForUser[i][1])
        AverageUser = np.mean(np.array(UserMovie),0)[1]

        if denom == 0:
            return AverageUser
        return AverageUser + (float(num)/float(denom))



    def run(self):
        sc = SparkContext("local[8]", "ratings")

        inputRDD = sc.textFile(self.inputFile)
        testRDD = sc.textFile(self.testFile)

        header = inputRDD.first() #extract header
        inputRDD = inputRDD.filter(lambda row : row != header)

        header2 = testRDD.first() #extract header
        testRDD = testRDD.filter(lambda row : row != header2)

        inputRDD = inputRDD.map(lambda line: line.split(',')).map(lambda x:  ((int(x[0]), int(x[1])),float(x[2])) )
        testRDD = testRDD.map(lambda line: line.split(',')).map(lambda x:  ((int(x[0]),int(x[1])),1))

        input = inputRDD.subtractByKey(testRDD)

        UserInput = input.map(lambda x: (x[0][0],(x[0][1], x[1]))).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
        self.UserDict = dict(UserInput.collect())
        print(len(self.UserDict.items()), len(UserInput.collect()))

        self.ComputeAllCorrelations()

        testRDD = testRDD.map(lambda x: ((x[0][0], x[0][1]), np.clip(self.predictValue(x[0][0],x[0][1]),0,5)))
        print(testRDD.take(2))

        WritingOutput = testRDD.map(lambda x: (x[0][0], (x[0][1],x[1]))).groupByKey().map(lambda x : (x[0], sorted(list(x[1]),key=lambda tup: tup[0]))).sortByKey().collect()

        with open('Tuhina_Kumar_UserBasedCF.txt','w') as f:
            for i in range(len(WritingOutput)):
                str1 =''
                for j in range(len(WritingOutput[i][1])):
                    str1 = str1 + str(WritingOutput[i][0]) +', ' + str(WritingOutput[i][1][j][0])+', '+ str(WritingOutput[i][1][j][1])+'\n'
                f.write(str1)


        error = testRDD.join(inputRDD)
        print('inputRDD join completed')
        MSE = error.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        print('MSE completed')
        RMSE = np.sqrt(MSE)
        print(RMSE)

        diff = error.map(lambda r: ( np.clip(np.floor(np.abs(r[1][0] - r[1][1])),0,4), 1)).reduceByKey(lambda a,b : a+b).sortByKey().collect()

        for i in range(len(diff)):
            print(diff[i])

        print('>=0 and <1:',diff[0][1])
        print('>=1 and <2:', diff[1][1])
        print('>=2 and <3:', diff[2][1])
        print('>=3 and <4:', diff[3][1])
        print('>=4', diff[4][1])








if __name__ == '__main__' :
    start = time.time()

    inputFile = sys.argv[1]
    testFile = sys.argv[2]

    CF = UserBasedCF(inputFile, testFile)
    CF.run()
    print(time.time() - start)
