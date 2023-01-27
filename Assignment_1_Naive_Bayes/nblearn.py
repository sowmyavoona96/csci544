import sys
import glob
import os
import json
import string


#stopwords = {"a", "the", "i", "at", "for", "of", "it", "its", "this", "my", "was", "is", "there", "has", 
#"been", "as", "in", "to", "we", "just", "be", "or", "their"}
stopwords = {}

labelIndex = {'truthful': 0, 'deceptive': 1, 'positive': 2, 'negative': 3}
priorProb = {'truthful': 0, 'deceptive': 0, 'positive': 0, 'negative': 0}
numLabels = 4
condProb = []
docCount = 0
trainData = {}
trainLabel1 = {}    #truthful, deceptive
trainLabel2 = {}    #positive, negative
vocabIndex = {}


def genClassifier(filepath):

    filePattern = '*/*/*/*.txt'

    fileList = glob.glob(os.path.join(filepath, filePattern))

    ind = 0
    for file in fileList:
        label1 = ''
        label2 = ''
        if("truthful" in file):
            label1 += 'truthful'
        elif("deceptive" in file): 
            label1 += 'deceptive'

        if("positive" in file):
            label2 = 'positive'
        elif("negative" in file):
            label2 = 'negative'
    
        priorProb[label1] += 1
        priorProb[label2] += 1

        content = open(file, "r").read()

        #get tokens after processing data
        tokens = processData(content)
       
        trainData[ind] = tokens
        trainLabel1[ind] = label1
        trainLabel2[ind] = label2
        ind+=1

    #update total count of documents in train set
    docCount = ind

    # calc prior probabilities
    for key in priorProb.keys():
        priorProb[key] = priorProb[key]/docCount

    condProb = calcCondProb()
    return numLabels, labelIndex, vocabIndex, priorProb, condProb
  
def calcCondProb():
    nTokensInLabel = [len(labelIndex)] * 4
    vocabCounter = 0
    for ind, tokens in trainData.items(): 
        for token in tokens: 
            if vocabIndex.get(token) is None:
                vocabIndex[token] = vocabCounter
                vocabCounter+=1
                condProb.append([1]* numLabels)
            
            tokenInd = vocabIndex[token]
            
            condProb[tokenInd][labelIndex[trainLabel1[ind]]]+= 1
            condProb[tokenInd][labelIndex[trainLabel2[ind]]]+= 1
            
            nTokensInLabel[labelIndex[trainLabel1[ind]]] += 1
            nTokensInLabel[labelIndex[trainLabel2[ind]]] += 1

    #print(f"vocab length: {len(vocabIndex)}")
    for i in range(len(vocabIndex)):
        for j in range(len(labelIndex)):
            condProb[i][j] = condProb[i][j]/(nTokensInLabel[j] + vocabCounter)

    return condProb

def stripPunc(content):
    ans = ''
    for i in range(0, len(content)):
        if content[i] in string.punctuation:
            # content = content[:i] + ' punc ' + content[i+1:]
            # ans += ' '
            if content[i] == "'":
                ans += ''
            else: ans += ' punc ' 
        else:
            ans+=content[i]

    return ans

def processData(content):
    content = content.lower()
    content = stripPunc(content)
    split = content.split(' ')
    tokens = []
    for token in split:
        token = token.strip()
        if(len(token) > 1 and token not in stopwords):
            tokens.append(token)
    return tokens

def writeModel(numLabels, labelIndex, vocabIndex, priorProb, condProb):
    print("writing to file: nbmodel.txt")
    file = open("nbmodel.txt", 'w')
    
    model = {}
    model["numLabels"] = numLabels
    model["labelIndex"] = labelIndex
    model["vocabIndex"] = vocabIndex
    model["priorProb"] = priorProb
    model["condProb"] = condProb
    json.dump(model, file, indent=2)

    
if __name__ == "__main__" :

    numLabels,labelIndex, vocabIndex, priorProb, condProb = genClassifier(sys.argv[1])
    writeModel(numLabels, labelIndex, vocabIndex, priorProb, condProb)