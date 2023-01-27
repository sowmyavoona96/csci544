import json
import sys
import math
import glob
import os
from nblearn import processData

numLabels = 4
labelIndex = {}
vocabIndex = {}
priorProb = {}
condProb = []
model = {}

def classify(filepath, numLabels, labelIndex, vocabIndex, priorProb, condProb):
    #print("classifying: " + filepath)
    content = open(filepath, "r").read()
    tokens = processData(content)
    scores = [0] * numLabels

    for label, labelInd in labelIndex.items():
        scores[labelInd] = math.log(priorProb[label])
        #print(f"label: {label}, ind:{labelInd}, prior: {priorProb[label]}")
        for token in tokens: 
            if token not in vocabIndex.keys(): 
                continue
            scores[labelInd] += math.log(condProb[vocabIndex[token]][labelInd])
    
    class1 = ''
    class2 = ''
   
    if scores[labelIndex['positive']] > scores[labelIndex['negative']]:
       class2 = 'positive'
    else: 
        class2 = 'negative'
    
    if scores[labelIndex['truthful']] > scores[labelIndex['deceptive']]:
       class1 = 'truthful'
    else: 
        class1 = 'deceptive'
   
    return class1, class2
        
def classifyFiles(folderpath):
    
    numLabels, labelIndex, vocabIndex, priorProb, condProb = readModel()
    
    filepattern = '*/*/*/*.txt'
    fileList = glob.glob(os.path.join(folderpath, filepattern))
   

    outputContent = ""


    for file in fileList:
        class1, class2  = classify(file, numLabels, labelIndex, vocabIndex, priorProb, condProb)
        outputContent += class1 + " " + class2 + " " + file + "\n" 
            

    outputFile = open("nboutput.txt", 'w')
    outputFile.write(outputContent)
    
def readModel():
    filepath = "nbmodel.txt"
    file = open(filepath, 'r')
    model = json.load(file)
    file.close()
    numLabels = model["numLabels"]
    labelIndex = model["labelIndex"]
    vocabIndex = model["vocabIndex"]
    priorProb = model["priorProb"]
    condProb = model["condProb"]
    
    return numLabels, labelIndex, vocabIndex, priorProb, condProb

if __name__ == "__main__" :
    classifyFiles(sys.argv[1])
  