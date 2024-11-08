import datetime
import time
import nltk
import json
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# global variables
SPANISH_STOP_WORDS = stopwords.words('spanish')
ALLOWED_POS = ["J"]

def findFeatures(reviewTokens, featureTokenList):
    # iterate through tokens to see if within token list
    words = set(reviewTokens)
    features = {}
    for w in featureTokenList:
        features[w] = (w in words)

    return features

def tokenizeReviewEnglish(review):
    # init our token list
    returnedTokens = []

    # split our review into tokens
    tokens = word_tokenize(review)

    # keep only adjectives
    posTokens = nltk.pos_tag(tokens)
    for currentToken in posTokens:
        if currentToken[1][0] in ALLOWED_POS:
            newToken = currentToken[0].lower()
            if len(newToken) > 1 and not newToken.isnumeric():
                returnedTokens.append(newToken)

    return returnedTokens

def tokenizeReviewSpanish(review, englishDict):
    # init our token list
    returnedTokens = []

    # split our review into tokens
    tokens = word_tokenize(review)

    # keep only words within the stored translation dictionary
    for currentToken in tokens:
        newToken = currentToken.lower()
        if newToken in englishDict:
            returnedTokens.append(englishDict[newToken])

    return returnedTokens


if __name__ == '__main__':

    startTime = time.time()

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------TRAINING-------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    print(f"*{datetime.datetime.now()}* Starting Reading Training Doc")

    # first we want to read in the full document
    rawReviewsTrain = []

    # Load JSON data from a file
    with open('prog4-movies/eng.imdb.train.jsonl', 'r', encoding='utf-8') as trainDataEnglishFile:
        for line in trainDataEnglishFile:
            rawReviewsTrain.append(json.loads(line))

    print(f"We have {len(rawReviewsTrain)} documents to train with")

    print(f"*{datetime.datetime.now()}* Starting Tokenization of Train")

    # read through the reviews to tokenize all our values
    allTokens = []
    for currentReviewClassify in rawReviewsTrain:
        allTokens.extend(tokenizeReviewEnglish(currentReviewClassify['text']))

    # sort the dictionary by its key values
    allTokensFreqDist = nltk.FreqDist(allTokens)

    # save down feature distributions
    rawFeatureFreqs = []
    for currentKey, currentCount in allTokensFreqDist.items():
        rawFeatureFreqs.append({
            "Token": currentKey,
            "Count": currentCount
        })

    # only return top 500 adjectives
    featureFreq = pd.DataFrame(rawFeatureFreqs)
    featureFreq = featureFreq.nlargest(1500, 'Count')

    # store our tables
    featureFreq.to_excel("featureFreq.xlsx", index=False)
    print("Feature freq results saved to 'featureFreq.xlsx'")

    # store grab our list of tokens
    featureTokens = featureFreq['Token'].to_list()

    # translate our top words into spanish dictionary
    transCounter = 1
    spanEngDict = {}
    for englishWord in featureTokens:
        spanishTranslation = GoogleTranslator(source='spanish', target='english').translate(englishWord)
        if spanishTranslation is not None:
            spanEngDict[spanishTranslation] = englishWord
            print(f"*{datetime.datetime.now()}* Translated word {transCounter}: {englishWord}")
        else:
            print(f"*{datetime.datetime.now()}* Did not translate word {transCounter}: {englishWord}")
        transCounter += 1

    print(f"*{datetime.datetime.now()}* Create Train Feature Set")

    # create our training feature set
    featureSetTrain = []
    for currentReviewTrain in rawReviewsTrain:
        relevantTokens = tokenizeReviewEnglish(currentReviewTrain['text'])
        featureCheck = findFeatures(relevantTokens, featureTokens)
        featureSetTrain.append((featureCheck, currentReviewTrain['label']))

    print(f"*{datetime.datetime.now()}* Create Classifier")

    naiveBayesClassifier = nltk.NaiveBayesClassifier.train(featureSetTrain)

    # -------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------TESTING--------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    print(f"*{datetime.datetime.now()}* Starting Reading Testing Doc")

    # first we want to read in the full document
    rawReviewsTest = []

    # Load JSON data from a file
    with open('prog4-movies/spa.muchocine.test.jsonl', 'r', encoding='utf-8') as testDataSpanishFile:
        for line in testDataSpanishFile:
            rawReviewsTest.append(json.loads(line))

    print(f"*{datetime.datetime.now()}* Create Test Feature Set")

    # create our training feature set
    featureSetTest = []
    runningListOfIDs = []
    counter = 1
    for currentReviewTest in rawReviewsTest:
        relevantTokens = tokenizeReviewSpanish(currentReviewTest['text'], spanEngDict)
        featureCheck = findFeatures(relevantTokens, featureTokens)
        featureSetTest.append((featureCheck, currentReviewTest['label']))
        runningListOfIDs.append(currentReviewTest['id'])

        print(f"*{datetime.datetime.now()}* Done Test {counter}")
        counter += 1

    # Create a list to store results
    rawResults = []
    condensedResults = []

    print(f"*{datetime.datetime.now()}* Iterating through Tests")

    # evaluate each individual test case and store in the results list
    print("Classifier accuracy percent:", (nltk.classify.accuracy(naiveBayesClassifier, featureSetTest))*100)
    for i, (currentFeatureTest, actualLabel) in enumerate(featureSetTest):
        predictedLabel = naiveBayesClassifier.classify(currentFeatureTest)
        rawResults.append({
            "Test Case": runningListOfIDs[i],
            "Features": currentFeatureTest,
            "Actual Label": actualLabel,
            "Predicted Label": predictedLabel
        })
        condensedResults.append({
            "Test Case": runningListOfIDs[i],
            "Actual Label": actualLabel,
            "Predicted Label": predictedLabel
        })

    # convert results list to DataFrame, save to csv
    rawResults = pd.DataFrame(rawResults)
    rawResults.to_csv("rawResultsSpain2.csv", index=False)
    print("Raw results saved to 'rawResultsSpain2.csv'")

    condensedResults = pd.DataFrame(condensedResults)
    condensedResults.to_excel("condResultsSpain2.xlsx", index=False)
    print("Condensed results saved to 'condResultsSpain2.xlsx'")

    endTime = time.time()

    elapsedTime = endTime - startTime

    print(f"*{datetime.datetime.now()}* Done Creation Program")
    print(f"Time to complete: {elapsedTime}")
