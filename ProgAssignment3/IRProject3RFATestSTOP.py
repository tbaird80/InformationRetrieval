import string
import numpy as np
import pandas as pd
import sys
import datetime
import time
import io
import struct
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

if __name__ == '__main__':

    startTime = time.time()

    # init our dictionaries to track our dictionary universes, plus document count
    docFreqCount = {}
    termFreqCount = {}
    postingsList = {}
    numberOfDocs = 0

    # create our list of stop words an initialize our porter stemmer
    nltk.download('stopwords')
    stopwordsList = set(stopwords.words('english'))
    ps = PorterStemmer()

    # read in the file name which is passed as a command line arg
    currentFileName = "rfa.txt"
    print(f"Reading in text file: {currentFileName}")

    print(f"*{datetime.datetime.now()}* Starting Doc Processing")
    # iterate through the open file line by line
    with open(currentFileName, 'r', encoding='utf-8') as fullFile:
        currentParagraph = []
        for rawNextLine in fullFile:
            if rawNextLine == '\n':
                pass
            else:
                nextLine = rawNextLine.replace('—', ' ')
                nextLine = nextLine.replace('/', ' ')
                nextLine = nextLine.replace('~', ' ')
                nextLine = nextLine.replace('.', ' ')
                nextLine = nextLine.replace('”', ' ')
                nextLine = nextLine.replace('’', ' ')
                nextLine = nextLine.replace('-', ' ')
                nextLine = nextLine.replace('?', ' ')
                nextLine = nextLine.replace(',', ' ')
                nextLine = nextLine.replace(';', ' ')
                nextLine = nextLine.replace(':', ' ')
                nextLine = nextLine.split()

                # pass if line empty
                if len(nextLine) == 0:
                    pass
                # update the document ID where relevant
                elif nextLine[0] in ('<P', '<Q'):
                    currentDocID = nextLine[1].lstrip('ID=')
                    currentDocID = int(currentDocID.rstrip('>'))
                    numberOfDocs += 1
                    # print(f"*{datetime.datetime.now()}* Processing Doc {currentDocID}")
                # hit end of paragraph, time to empty out the tokens
                elif rawNextLine in ('</P>\n', '</Q>\n'):
                    tempDict = {}
                    for currentToken in currentParagraph:
                        # skip blanks
                        if currentToken == "":
                            pass
                        elif currentToken in tempDict:
                            tempDict[currentToken] += 1
                        else:
                            tempDict[currentToken] = 1

                    for token, count in tempDict.items():
                        if token in docFreqCount:
                            docFreqCount[token] += 1
                            termFreqCount[token] += count
                            postingsList[token].append((currentDocID, count))
                        else:
                            docFreqCount[token] = 1
                            termFreqCount[token] = count
                            postingsList[token] = [(currentDocID, count)]
                    currentParagraph = []
                # process another line of text
                else:
                    nextLine = [word.lower() for word in nextLine]
                    nextLine = [word.strip() for word in nextLine]
                    nextLine = [word.strip(string.punctuation) for word in nextLine]
                    #nextLine = [ps.stem(word) for word in nextLine]
                    currentParagraph.extend(nextLine)

    print(f"*{datetime.datetime.now()}* Done Doc Processing")

    print(f"*{datetime.datetime.now()}* Start stop word removal")
    # remove stop words from list of possible values
    for currentStopWord in stopwordsList:
        if currentStopWord in docFreqCount:
            del docFreqCount[currentStopWord]
            del termFreqCount[currentStopWord]
            del postingsList[currentStopWord]

    print(f"*{datetime.datetime.now()}* End stop word removal")

    # Create our dictionary table with all relevant information
    docFreqTable = pd.DataFrame(list(docFreqCount.items()), columns=['Term', 'DocFreq']).set_index('Term')
    termFreqTable = pd.DataFrame(list(termFreqCount.items()), columns=['Term', 'TermFreq']).set_index('Term')
    postingsListTable = pd.DataFrame(list(postingsList.items()), columns=['Term', 'PostingList']).set_index('Term')
    fullFreqTable = docFreqTable.join(termFreqTable)
    fullFreqTable['IDF'] = np.log2(numberOfDocs/fullFreqTable['DocFreq'])
    fullFreqTable = fullFreqTable.join(postingsListTable)
    fullFreqTable = fullFreqTable.sort_index()

    print(f"*{datetime.datetime.now()}* Starting Binary Stream Processing")

    # create binary stream that will account for 4 bytes for each doc id, term freq pair
    numBytesNeeded = fullFreqTable['DocFreq'].sum() * 4 * 2
    buffer = bytearray(numBytesNeeded)

    # create way to track the offset of the given value
    currentByte = 0
    fullFreqTable['Offset'] = [0] * len(fullFreqTable)

    # iterate through the indices in order
    for currentTerm in fullFreqTable.index:

        # get postings list and create offset value
        currentPostingsList = postingsList[currentTerm]
        fullFreqTable.loc[currentTerm, 'Offset'] = currentByte

        # iterate through all pairs within the given term
        for currentPair in currentPostingsList:
            # write to the binary stream
            struct.pack_into('i', buffer, currentByte, currentPair[0])
            currentByte += 4
            struct.pack_into('i', buffer, currentByte, currentPair[1])
            currentByte += 4

    print(f"*{datetime.datetime.now()}* Done Binary Stream Processing")

    # print information about dictionary
    vocabSize = len(fullFreqTable)
    collectionSize = fullFreqTable['TermFreq'].sum()
    print(f"Number of Documents: {numberOfDocs}")
    print(f"Vocab Size: {vocabSize}")
    print(f"Collection Size: {collectionSize}")

    # our file names for output
    currentFileNameMinusTxt = currentFileName.rstrip('.txt')
    dictFile = "rfaDict.csv"
    invertedFile = "rfaInvertedFile"

    # drop the excess columns for the actual dictionary
    fullFreqTable.drop(['TermFreq', 'PostingList'], axis=1, inplace=True)
    fullFreqTable.to_csv(dictFile, index=True)

    # write binary file to disk
    binaryStream = io.BytesIO(buffer)
    with open(invertedFile, "wb") as f:
        f.write(binaryStream.read())

    endTime = time.time()

    elapsedTime = endTime - startTime

    print(f"*{datetime.datetime.now()}* Done Creation Program")
    print(f"Time to complete: {elapsedTime}")
