from pyspark import SparkContext
import os
from itertools import groupby, combinations
from operator import add
import time
import random
import sys
import numpy as np

class LSH:
    def __init__(self, numHash, numBand):
        self.numHash = numHash
        self.numBand = numBand
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



    def run(self, inputFile):
        sc = SparkContext("local[8]", "ratings")
        text = sc.textFile(inputFile)
        header = text.first() #extract header
        text = text.filter(lambda row : row != header)

        mapping = text.map(lambda line: line.split(',')).map(lambda x:  (int(x[0]),int(x[1]))).groupByKey()
        movieInput = text.map(lambda line: line.split(',')).map(lambda x:  (int(x[1]),int(x[0]))).groupByKey().map(lambda x: (x[0],list(x[1]))).sortByKey()
        self.totalUsers = len(mapping.collect())
        print(self.totalUsers)
        self.generate_hash_functions()

        #for i in range(10):
        #   print(movieInput.collect()[i])

        Signature = movieInput.map(lambda x : (x[0],self.create_signature(x[1])))
        #print("Signature")
        #print(Signature.take(10))


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
        print("flatmap",time.time()- start)
        unique = unique.distinct().map(lambda x :(x[0], (x[1] ,self.computeJacardSimilarity(x[0],x[1])))).filter(lambda x : x[1][1] >= 0.5)
        output = unique.groupByKey().sortByKey().map(lambda x: (x[0], sorted(list(x[1]),key=lambda tup: tup[0]))).collect()
        print("done",time.time()- start)

        with open('Tuhina_Kumar_SimilarMovie_Jaccard.txt','w') as f:
            for i in range(len(output)):
                str1 =''
                for j in range(len(output[i][1])):
                    str1 = str1 + str(output[i][0]) +', ' + str(output[i][1][j][0])+', '+ str(output[i][1][j][1])+'\n'
                f.write(str1)
        #print("str done",time.time()- start)


        ##### precision and recall ########
        '''
        Ground_truth = sc.textFile('data/SimilarMovies.GroundTruth.05.csv')
        header = Ground_truth.first() #extract header
        Ground_truth = Ground_truth.filter(lambda row : row != header).map(lambda line: line.split(',')).map(lambda x:  (int(x[0]),int(x[1]))).collect()
        g = set(Ground_truth)
        t = set(unique.map(lambda x: (x[0],x[1][0])).collect())

        tp = len(g & t)
        fp = len(t - g)
        fn = len(g - t)

        pr = float(tp)/float(tp+fp)
        re = float(tp)/float(tp + fn)

        print(" precision", pr, "recall",re)
        '''



if __name__ == '__main__':
    start = time.time()
    inputFile = sys.argv[1]
    numHash = 20

    l = LSH(20,5)
    l.run(inputFile)
    print("total_time",time.time() - start )
