import datetime
import time
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# global variables
SPANISH_STOP_WORDS = stopwords.words('spanish')

def findFeatures(reviewTokens, featureTokenList):
    # iterate through tokens to see if within token list
    words = set(reviewTokens)
    features = {}
    for w in featureTokenList:
        features[w] = (w in words)

    return features

def tokenizeReview(review):
    # init our token list
    returnedTokens = []

    # split our review into tokens
    tokens = word_tokenize(review)

    for currentToken in tokens:
        newToken = currentToken.lower()
        if newToken not in SPANISH_STOP_WORDS and len(newToken) > 1 and not newToken.isnumeric():
            returnedTokens.append(newToken)

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
    with open('prog4-movies/spa.muchocine.train.jsonl', 'r', encoding='utf-8') as trainDataSpanishFile:
        for line in trainDataSpanishFile:
            rawReviewsTrain.append(json.loads(line))

    print(f"We have {len(rawReviewsTrain)} documents to train with")

    print(f"*{datetime.datetime.now()}* Starting Tokenization of Train")

    # read through the reviews to tokenize all our values
    allTokens = []
    for currentReviewClassify in rawReviewsTrain:
        allTokens.extend(tokenizeReview(currentReviewClassify['text']))

    # sort the dictionary by its key values
    allTokensFreqDist = nltk.FreqDist(allTokens)

    # grab all values over a specific size
    print(f"*{datetime.datetime.now()}* Starting Organizing Tokens by Freq")

    # save down feature distributions
    rawFeatureFreqs = []
    for currentKey, currentCount in allTokensFreqDist.items():
        rawFeatureFreqs.append({
            "Token": currentKey,
            "Count": currentCount
        })

    featureFreq = pd.DataFrame(rawFeatureFreqs)
    featureFreq = featureFreq[featureFreq['Count'] > 40]

    featureFreq.to_excel("featureFreqSpanish1.xlsx", index=False)
    print("Feature freq results saved to 'featureFreqSpanish1.xlsx'")

    featureTokens = featureFreq['Token'].to_list()

    print(f"*{datetime.datetime.now()}* Create Train Feature Set")

    # create our training feature set
    featureSetTrain = []
    for currentReviewTrain in rawReviewsTrain:
        relevantTokens = tokenizeReview(currentReviewTrain['text'])
        featureCheck = findFeatures(relevantTokens, featureTokens)
        featureSetTrain.append((featureCheck, currentReviewTrain['label']))

    print(f"*{datetime.datetime.now()}* Create Classifier")

    naiveBayesClassifier = nltk.NaiveBayesClassifier.train(featureSetTrain)

    print(naiveBayesClassifier.show_most_informative_features(100))

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
    for currentReviewTest in rawReviewsTest:
        relevantTokens = tokenizeReview(currentReviewTest['text'])
        featureCheck = findFeatures(relevantTokens, featureTokens)
        featureSetTest.append((featureCheck, currentReviewTest['label']))
        runningListOfIDs.append(currentReviewTest['id'])

    # Create a list to store results
    rawResults = []
    condensedResults = []

    print(f"*{datetime.datetime.now()}* Iterating through Tests")

    # evaluate each individual test case and store in the results list
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
    rawResults.to_csv("rawResultsSpain1.csv", index=False)
    print("Raw results saved to 'rawResultsSpain1.csv'")

    condensedResults = pd.DataFrame(condensedResults)
    condensedResults.to_excel("condResultsSpain1.xlsx", index=False)
    print("Condensed results saved to 'condResultsSpain1.xlsx'")

    endTime = time.time()

    elapsedTime = endTime - startTime

    print(f"*{datetime.datetime.now()}* Done Creation Program")
    print(f"Time to complete: {elapsedTime}")
