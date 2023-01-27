import sys
import glob
import os
import json
import string
import math
from pprint import pprint

def hmmDecode(testFile):
    tagDict, tagIndexMap, vocabDict, initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts = readModel()
    #print(f"decode: tag size: {len(tagDict)}, tagIndex size: {len(tagIndexMap)}, word size: {len(vocabDict)}")
    outputContent = ""
    sentences = open(testFile, "r", encoding='utf-8').readlines()
    for sentence in sentences:
        tags  = viterbi(sentence, tagDict, tagIndexMap, vocabDict, initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts)
        #print(f"path: {tags}")
        taggedSentence = getTaggedSentence(sentence, tags)
        #print(f"tagse: {taggedSentence}")
        outputContent += taggedSentence + "\n"      
    output(outputContent)

def viterbi(sentence, tagDict, tagIndexMap, vocabDict, initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts):
    words = sentence.split()
    viterbiMatrix = [[0] *len(words) for _ in range(len(tagDict))]
    backPointer = [[0] *len(words) for _ in range(len(tagDict))]
    bestPath = ""

    #count new words in sentence
    newWords = 0
    for word in words:
        if(vocabDict.get(word) is None):
            newWords += 1

    #initialize viterbi matrix for all rows for first column
    for i, (tagIndex, tag) in enumerate(tagIndexMap.items()):
        viterbiMatrix[tagIndex][0] = initProbMatrix[tagIndex]

        #todo: in case of unseen word: ignore emission prob?
        if(vocabDict.get(words[0]) is not None):
            viterbiMatrix[tagIndex][0]  *= emissionMatrix[tagIndex][vocabDict[words[0]][0]]
        else:
            viterbiMatrix[tagIndex][0] *= tagVsWordCounts[tagIndex][1]/(newWords * tagVsWordCounts[tagIndex][0])
    
    #forward propagation
    for j in range(1, len(words)):
        word = words[j]
        for i, (tagIndex, tag) in enumerate(tagIndexMap.items()):
           
            #for unseen word: emission prob is 1
            emissionProb = 1
            if(vocabDict.get(word) is None):
                emissionProb = tagVsWordCounts[tagIndex][1]/(newWords * tagVsWordCounts[tagIndex][0])
                #todo emissionProb
            else:
                #when emission from tag to word has not been seen in train data, ignore the current state
                if(emissionMatrix[tagIndex][vocabDict[word][0]] == 0):
                  continue
                else:
                    emissionProb = emissionMatrix[tagIndex][vocabDict[word][0]]
    
            for k, (prevTagIndex, prevTag) in enumerate(tagIndexMap.items()):
                if(viterbiMatrix[prevTagIndex][j-1] is None):
                    print(f"viterbmatrix none for: {word}, {tag}, {prevTag}")
                    continue
                currProb = viterbiMatrix[prevTagIndex][j-1] * (emissionProb)  * (transitionMatrix[prevTagIndex][tagIndex])
                if(viterbiMatrix[tagIndex][j] is None 
                or currProb > viterbiMatrix[tagIndex][j]):
                    viterbiMatrix[tagIndex][j] = currProb
                    backPointer[tagIndex][j] = prevTagIndex

    tags = getTags(words, backPointer, viterbiMatrix, tagIndexMap)
    return tags

def getTags(words, backPointer, viterbiMatrix, tagIndexMap):
    lastWordTagIndex = 0
    lastTagBestProb = viterbiMatrix[lastWordTagIndex][len(words)-1]
    for i, (tagIndex, tag) in enumerate(tagIndexMap.items()):
        if(lastTagBestProb is None and viterbiMatrix[tagIndex][len(words)-1] is not None):
            lastTagBestProb = viterbiMatrix[tagIndex][len(words)-1] 
            lastWordTagIndex = tagIndex
        
        if(viterbiMatrix[tagIndex][len(words)-1] is not None 
            and  viterbiMatrix[tagIndex][len(words)-1] > lastTagBestProb):
            lastTagBestProb = viterbiMatrix[tagIndex][len(words)-1]
            lastWordTagIndex = tagIndex
    
    #print(f"last word tag with: {lastWordTagIndex}, {tagIndexMap[lastWordTagIndex]}")
    currTagIndex = lastWordTagIndex
    bestPath = []
    bestPath.append(str(tagIndexMap[currTagIndex]))
    for i in reversed(range(0, len(words)-1)):
        currTagIndex = backPointer[currTagIndex][i+1]
        bestPath.append(str(tagIndexMap[currTagIndex]))
       # print(f"i: {i} tag with: {currTagIndex}, {tagIndexMap[currTagIndex]}")
       
    bestPath.reverse()
    return bestPath

def getTaggedSentence(sentence, tags):
    output = ""
    words = sentence.split()
    for i, word in enumerate(words):
        output += word + "/" + tags[i]+ " "
    return output[:-1]

def output(outputContent):
    file = open("hmmoutput.txt", "w", encoding='utf-8')
    file.write(outputContent)
    file.close()

def readModel():
    filepath = "hmmmodel.txt"
    file = open(filepath, 'r', encoding='utf-8')
    model = json.load(file)
    file.close()
    tagDict = model["tagDict"]
    tagIndexMap = model["tagIndexMap"]
    vocabDict = model["vocabDict"] 
    initProbMatrix = model['initProbMatrix']
    emissionMatrix = model["emissionMatrix"]
    transitionMatrix = model["transitionMatrix"]
    tagVsWordCounts = model["tagVsWordCounts"]
    tagVsWordCounts = {int(k):v for k,v in tagVsWordCounts.items()}
    tagIndexMap = {int(k):v for k,v in tagIndexMap.items()}
    return tagDict, tagIndexMap, vocabDict, initProbMatrix, emissionMatrix, transitionMatrix, tagVsWordCounts

if __name__ == "__main__" :

    hmmDecode(sys.argv[1])
    #Write model