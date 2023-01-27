from pickle import TRUE
import sys
import glob
import os
import json
import string
from pprint import pprint 
import math

def writeModel(tagDict, tagIndexMap, vocabDict, initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts):
    #print("writing to file: hmmmodel.txt")
    file = open("hmmmodel.txt", 'w', encoding='utf-8')
    
    model = {}
    model["tagDict"] = tagDict #tag vs (index, count)
    model["tagIndexMap"] = tagIndexMap #index vs tag
    model["vocabDict"] = vocabDict #word vs (index, count)
    model['initProbMatrix'] = initProbMatrix # index is tag index
    model["emissionMatrix"] = emissionMatrix #rows: tags, columns: words
    model["transitionMatrix"] = transitionMatrix #rows: prev tag ti-1, column: ti
    model["tagVsWordCounts"] = tagVsWordCounts

    json.dump(model, file, indent=4)

def hmmLearn(filepath):

    taggedSentences = open(filepath, "r", encoding = 'utf-8').readlines()
    tagDict, tagIndexMap, vocabDict = getDicts(taggedSentences)
    #print(f"learn: tag size: {len(tagDict)}, tagIndex size: {len(tagIndexMap)}, word size: {len(vocabDict)}")
    
    initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts = calcMatrix(tagDict, tagIndexMap, vocabDict, taggedSentences)
    #print(f"emissionMatrix size: {len(emissionMatrix)}, transitionMatrix size: {len(transitionMatrix)}")

    writeModel(tagDict, tagIndexMap, vocabDict, initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts)

def getDicts(taggedSentences):
    tagDict = {}
    vocabDict = {}
    tagIndexMap = {}
    tagVsWordSet = {}
    tagIndex = 0
    vocabIndex = 0
    for sentence in taggedSentences:
        taggedWords = sentence.split()
        prevTag = '_START_'
        for taggedWord in taggedWords:
            word, tag = split(taggedWord)

            if(tagDict.get(tag) is None):
                tagDict[tag] = (tagIndex, 0)
                tagIndexMap[tagIndex] = tag
                tagIndex += 1
                tagVsWordSet[tag] = set()
            tagDict[tag] = (tagDict[tag][0], tagDict[tag][1] + 1)
            tagVsWordSet[tag].add(word)
            if(vocabDict.get(word) is None):
                vocabDict[word] = (vocabIndex, 0)
                vocabIndex += 1
            vocabDict[word] = (vocabDict[word][0], vocabDict[word][1] + 1)

    #print(f"tag vs word: {len(tagVsWordSet)}")
    return tagDict, tagIndexMap, vocabDict

def calcMatrix(tagDict, tagIndexMap, vocabDict, taggedSentences):
    emissionMatrix = [[0]* len(vocabDict) for _ in range(len(tagDict))]
    transitionMatrix = [[1] * len(tagDict) for _ in range(len(tagDict))]
    initProbMatrix = [1] * len(tagDict) #todo

    for taggedSentence in taggedSentences:
        taggedWords = taggedSentence.split()
        prevTag = '_START_'
        for taggedWord in taggedWords:
            word, tag = split(taggedWord)
           
            if(tagDict.get(tag) is None): continue #todo

            emissionMatrix[tagDict[tag][0]][vocabDict[word][0]] += 1

            if prevTag != '_START_':
                transitionMatrix[tagDict[prevTag][0]][tagDict[tag][0]] += 1
            else:
                initProbMatrix[tagDict[tag][0]] += 1
            prevTag = tag

    tagVsWordCounts = getTagVsWordCount(emissionMatrix)

    for i in range(len(initProbMatrix)):
        initProbMatrix[i] = initProbMatrix[i]/(len(taggedSentences) + len(tagDict)) #tagDict[tagIndexMap[i]][1] #(len(taggedSentences) + len(tagDict))

    for i in range(len(emissionMatrix)):
        for j in range(len(emissionMatrix[0])):
           emissionMatrix[i][j] = emissionMatrix[i][j]/(tagDict[tagIndexMap[i]][1])

    for i in range(len(transitionMatrix)):
        for j in range(len(transitionMatrix[0])):
            transitionMatrix[i][j] = transitionMatrix[i][j]/(tagDict[tagIndexMap[i]][1] + len(tagDict))
    # print("init")
    # print(initProbMatrix)
    # print("emission")
    # print(emissionMatrix)
    # print("transition")
    # print(transitionMatrix)
    return initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts

def getTagVsWordCount(emissionMatrix):
    tagVsWordCounts = {}
    for i in range(len(emissionMatrix)):
        count = 0
        uniqCount = 0
        for j in range(len(emissionMatrix[0])):
            if(emissionMatrix[i][j] == 1):
                uniqCount += 1
            count += emissionMatrix[i][j]
        tagVsWordCounts[i] = (count, uniqCount)
    
    #pprint(tagVsWordCounts)
    return tagVsWordCounts

def split(wordAndTag):
    split = wordAndTag.rsplit('/', 1)
    word = split[0]
    tag = split[1]
    return word, tag

if __name__ == "__main__" :

    hmmLearn(sys.argv[1])
    #Write model