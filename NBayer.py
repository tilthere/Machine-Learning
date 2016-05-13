# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:50:12 2016

Project: Project_2

@author: Xiaomei Huang

Version 3
"""
#==============================================================================
# This program uses Naive Bayes algorithm to perform classifcation: 
# Given a collection of trainning data consisting of category labeled documents
# learns how to classify new documents into the correct category
# Finally give out the learning accuracy. 
#==============================================================================


import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import timeit

# get all the training documents and their known classes
# The tags will be used for counting class probability
def loadData(filename):
    with open(filename) as f:
        lines = f.readlines()
    docs = []
    tags = []
    for i in range(len(lines)):
        docs.append(lines[i].split())
        tags.append(docs[i][0])
    return docs,tags
    
train_docs,train_tags = loadData('forumTraining.data')
 
# Get all the distinct vocabularies from training documents.
# Using Counter function to get the dictionary, then the keys are distinct vocs
# This function return dictionary data
def getVocs(filename,timesFilter):
    stop = stopwords.words('english')
    with open(filename) as f:
        words = f.read().split() 
    vocs = Counter(words)
    vocs = dict((w,c) for w,c in vocs.items() if (c> timesFilter) and (w not in stop))
    return vocs
    
#vocs = list(getVocs('forumTraining.data',0).keys())
#len(vocs)
#Out[5]: 73713 
   
#==============================================================================
#  Noticed here if we don't do the feature selection, it ends up to 73713 vocs
#  When looking close to those unfiltered words, we could see: 
# vocs[1:10]
# Out[8]: 
# ['aa',
#  'aaa',
#  'aaaa',
#  'aaaaaaaaaaaa',
#  'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaauuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuugggggggggggggggg',
#  'aaaaagggghhhh',
#  'aaaarrgghhhh',
#  'aaah',
#  'aaahh'] 
#  There are too many garbage words 
#  and they would cause much longer time for computer to process (Even crash)  
#  So features filtering is very import here
#  Those vocabularies which appeared less than 3 times and stop words such as 'the' are removed
# 
#==============================================================================
vocs = list(getVocs('forumTraining.data',3).keys())
len(vocs)
#Out[12]: 29384    
#Now we get 29384 words for training.


# Textj ← create a single document per class (concatenate all Docsj)
# Use dictionary structure, keys for class, values for text_j
def getVocDict(docList):
     vocDict= {}
     vocDict[docList[0][0]]=docList[0]
     # concatenate all the docs by the categories
     for i in range(1,len(docList)):
         if docList[i][0] == docList[i-1][0]:
             vocDict[docList[i][0]]+=(docList[i])  #Do not use append
         else:
             vocDict[docList[i][0]] = docList[i]
     return vocDict

train_texts = getVocDict(train_docs)
#len(train_texts)
#Out[16]: 20
#Get 20 categories texts


# Get the vocabulary matrix. Use Counter to get the total number of word position in test_j
# Add 1 to each position
# Use pandas dataframe structure
def vocMatrix(docs, vocab):
    matrix = []
    for key,doc in docs.items():
        counter = Counter(docs[key])
        row = [counter.get(w, 0) for w in vocab]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    temp = [1]*len(vocab)
    vMat = df.add(temp,axis = 1)
    vMat.columns = vocab
    return vMat

trainMat = vocMatrix(train_texts,vocs)
# get the row name for matrix
rowName = list(train_texts.keys())
#trainMat.shape
#Out[162]: (20, 29384)


    
# Probability estimate of a particular class:
# return dictionary structure
def classFreq(taglist):
    clsFreq = {}
    tagDict = Counter(taglist)
    total = sum(tagDict.values())
    for key, value in tagDict.items():
        clsFreq[key]= value/ total
    return clsFreq

clsFreq = classFreq(train_tags)

#==============================================================================
# clsFreq
# Out[21]: 
# {'atheism': 0.042504206145399806,
#  'autos': 0.05259895510493226,
#  'baseball': 0.05286460639334101,
#  'christianity': 0.05295315682281059,
#  'cryptology': 0.052687505534401845,
#  'electronics': 0.05233330381652351,
#  'forsale': 0.05180200123970601,
#  'graphics': 0.05171345081023643,
#  'guns': 0.048259984060922696,
#  'hockey': 0.053130257681749754,
#  'mac': 0.051182148233418934,
#  'medicine': 0.05259895510493226,
#  'mideastpolitics': 0.04994244222084477,
#  'motorcycles': 0.05295315682281059,
#  'mswindows': 0.05065084565660143,
#  'pc': 0.05224475338705393,
#  'politics': 0.041175949703356064,
#  'religion': 0.03338351191003276,
#  'space': 0.052510404675462675,
#  'xwindows': 0.052510404675462675}
# 
#==============================================================================


# Construct the frequency matrix
# To avoid data underflow, according to log(a*b)= log(a)+log(b)
# Take log to every elements in the matrix

deNom = trainMat.sum(axis=1)     # sum of each row
freqMat = trainMat.loc[:,:].div(deNom, axis=0)
freqLog = np.log(freqMat)

# Also construct classes frequency list and use log values.
freqList = []
for key,v in train_texts.items():
    freqList.append(clsFreq[key])
freqListLog = np.log(freqList)
#==============================================================================
# freqListLog
# Out[23]: 
# array([-2.93500869, -3.39969316, -2.94337694, -3.15815224, -2.94674394,
#        -2.95181581, -3.03115255, -2.94002123, -2.99688409, -2.98279935,
#        -2.96203736, -2.97236448, -2.93834759, -2.94505902, -2.93834759,
#        -2.94505902, -2.95012233, -2.9603265 , -3.18990094, -2.94674394])
# 
#==============================================================================


# <Classify>

# Get test documents
# Will use the tags to verify the result
test_docs,test_tags = loadData("ForumTest.data")

# Construct the test document matrix according to training vocabularies.
# Using pandas dataframe stucture
def countWord(doc,vocab):
    matrix = []
    counter = Counter(doc)
    row = [counter.get(w, 0) for w in vocab]
    for i in range(20):
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = vocab
    return df

# This function returns the class name
def clsfy(testdoc):
    test = countWord(testdoc,vocs)
    testmat = freqLog * test
    summ = testmat.sum(axis = 1)
    reslt = list(summ + freqListLog)
    pos = reslt.index(max(reslt))
    print("Tested category: " , rowName[pos])
    return rowName[pos]


# This function is to verify the classify result
def testResult(testdocslist,testtimes):
    reslt = [] 
    for i in range(testtimes):
        rand= int(np.random.uniform(0,len(testdocslist))) 
        print("Document index: ", rand)
        cls = clsfy(testdocslist[rand])
        print("Actual category:  {} \n" . format(testdocslist[rand][0]))
        if cls == testdocslist[rand][0]:
            reslt.append(1)
        else:
            reslt.append(0)
    accu = sum(reslt)/len(reslt) * 100
    print("Result for {} times guess (1 for correct guess, 0 for missed guess): \n {}".format(testtimes,reslt))
    print("\nThe accuracy with {} features is: {}% ".format(len(vocs),accu))
 


start = timeit.default_timer()
testResult(test_docs, 10)
stop = timeit.default_timer()
print ("Runing time:",(stop - start))



