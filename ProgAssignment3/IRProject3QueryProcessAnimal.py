import string
import pandas as pd
import datetime
import struct
import numpy as np
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

if __name__ == '__main__':
    startTime = time.time()

    #---------------------------------------------------------------------------------------------------------------
    #------------------------Document Vector Length Processing------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------

    print(f"*{datetime.datetime.now()}* Starting Query Processing")

    # our file names
    dictFile = "animalDict.csv"
    invertedFilePath = "animalInvertedFile"

    # read dictionary from file and find the total size of the inverted file using last item on list
    fullFreqTable = pd.read_csv(dictFile)

    print(f"*{datetime.datetime.now()}* Starting Document Vector Length Processing")
    print(f"*{datetime.datetime.now()}* Starting Document Vector Length Processing")


    # open binary file
    with open(invertedFilePath, 'rb') as invertedFile:
        # start at the beginning of the file
        inverseFileContent = invertedFile.read()

    # convert the binary file into list of integers
    inverseFileContent = struct.unpack(f'{len(inverseFileContent)//4}i', inverseFileContent)

    # create a dataframe out of the list, evens are doc ids and odds are doc freq
    docIDs = inverseFileContent[::2]
    docTermFreqs = inverseFileContent[1::2]

    print(f"*{datetime.datetime.now()}* Checkpoint: finished binary file processing")

    # create parallel list of terms that match the number of postings for each
    termList = fullFreqTable['Term'].to_list()
    termDFs = fullFreqTable['DocFreq'].to_list()
    termListExpanded = [value for value, freq in zip(termList, termDFs) for _ in range(freq)]

    # create our table with our values, merge in our IDF
    DocFreqByTerm = pd.DataFrame({'Term': termListExpanded, 'DocID': docIDs, 'DocTermFreq': docTermFreqs})
    DocFreqByTerm = pd.merge(DocFreqByTerm, fullFreqTable[['Term', 'IDF']], on='Term')

    print(f"*{datetime.datetime.now()}* Checkpoint: finished full table of document frequencies by term")

    # calc our weight
    DocFreqByTerm['CorpusTFIDF'] = DocFreqByTerm['DocTermFreq'] * DocFreqByTerm['IDF']
    DocFreqByTerm['SquaredDocVectorLength'] = DocFreqByTerm['CorpusTFIDF']**2
    DVITableAgg = DocFreqByTerm[['DocID', 'SquaredDocVectorLength']].groupby('DocID').agg('sum')
    DVITableAgg['DocVectorLength'] = np.sqrt(DVITableAgg['SquaredDocVectorLength'])

    print(f"*{datetime.datetime.now()}* Checkpoint: finished aggregated table of document vector length summary")

    # write our document vector lengths to file
    docVectorLengthFile = 'animalDocVectorLength.csv'
    DVITableAgg.to_csv(docVectorLengthFile, index=True)

    print(f"*{datetime.datetime.now()}* Done Document Vector Length Processing")

    # ---------------------------------------------------------------------------------------------------------------
    # ------------------------Query Processing-----------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------

    # init our dictionaries to track our dictionary universes, plus document count
    # our output table that we want to track as we go
    outputRankedList = pd.DataFrame(columns=['QueryID', 'Q0', 'DocID', 'Rank', 'CosineSimScore', 'JHID'])
    numberOfDocs = 0

    # create our list of stop words an initialize our porter stemmer
    nltk.download('stopwords')
    stopwordsList = set(stopwords.words('english'))
    ps = PorterStemmer()

    # read in the file name which is passed as a command line arg
    currentFileName = "animalQuery.txt"
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
                nextLine = nextLine.replace('"', ' ')
                nextLine = nextLine.replace("'", ' ')
                nextLine = nextLine.replace('-', ' ')
                nextLine = nextLine.replace('–', ' ')
                nextLine = nextLine.replace('?', ' ')
                nextLine = nextLine.replace(',', ' ')
                nextLine = nextLine.replace(';', ' ')
                nextLine = nextLine.replace(':', ' ')
                nextLine = nextLine.replace('|', ' ')
                nextLine = nextLine.replace('*', ' ')
                nextLine = nextLine.replace('&', ' ')
                nextLine = nextLine.replace('=', ' ')
                nextLine = nextLine.replace('(', ' ')
                nextLine = nextLine.replace(')', ' ')
                nextLine = nextLine.replace('%', ' ')
                nextLine = nextLine.replace('+', ' ')
                nextLine = nextLine.split()

                # pass if line empty
                if len(nextLine) == 0:
                    pass
                # update the document ID where relevant
                elif nextLine[0] in ('<P', '<Q'):
                    currentDocID = int(nextLine[2].rstrip('>'))
                    numberOfDocs += 1
                    # print(f"*{datetime.datetime.now()}* Processing Doc {currentDocID}")
                # hit end of paragraph, time to empty out the tokens
                elif rawNextLine in ('</P>\n', '</Q>\n'):
                    # remove stop words from list of possible values
                    for currentStopWord in stopwordsList:
                        if currentStopWord in currentParagraph:
                            currentParagraph.remove(currentStopWord)

                    tempDict = {}
                    for currentToken in currentParagraph:
                        # skip blanks, single characters and numbers
                        if currentToken == "" or len(currentToken) == 1 or currentToken.isnumeric():
                            pass
                        elif currentToken in tempDict:
                            tempDict[currentToken] += 1
                        else:
                            tempDict[currentToken] = 1
                    # save down unique terms in list before resetting
                    queryTermList = set(currentParagraph)
                    currentParagraph = []

                    # create table from term freq dictionary
                    QueryTermTable = pd.DataFrame(list(tempDict.items()), columns=['Term', 'QueryTermFreq'])
                    QueryTermTable.set_index('Term', inplace=True)
                    QueryTermTable['QueryID'] = [currentDocID] * len(QueryTermTable)

                    # get our corpus term/document freq table, subsetted by the terms from the given query
                    DocFreqByTermCurrentQuery = DocFreqByTerm[DocFreqByTerm['Term'].isin(queryTermList)]
                    DocFreqByTermCurrentQuery.set_index('Term', inplace=True)

                    # merge our corpus information with the information from query
                    DocFreqByTermCurrentQuery = DocFreqByTermCurrentQuery.join(QueryTermTable, how='inner')
                    DocFreqByTermCurrentQuery['QueryTFIDF'] = DocFreqByTermCurrentQuery['QueryTermFreq'] * DocFreqByTermCurrentQuery['IDF']

                    # calc query vector length
                    currentQueryVectorLengthTable = DocFreqByTermCurrentQuery.reset_index()
                    currentQueryVectorLengthTable = currentQueryVectorLengthTable[['Term', 'QueryTFIDF']].drop_duplicates()
                    currentQueryVectorLengthTable['SquaredDocVectorLength'] = currentQueryVectorLengthTable['QueryTFIDF'] ** 2
                    currentQueryVectorLength = np.sqrt(currentQueryVectorLengthTable['SquaredDocVectorLength'].sum())

                    # print the results from the first query
                    if currentDocID == 763:
                        currentQueryVectorLengthTable[['Term', 'QueryTFIDF']].to_csv('Query763Weights.txt', sep=' ', header=True, index=False)

                    # find dot product, sum the results as numerator of sim score
                    DocFreqByTermCurrentQuery['QueryDotProduct'] = DocFreqByTermCurrentQuery['QueryTFIDF'] * DocFreqByTermCurrentQuery['CorpusTFIDF']
                    QuerySimScore = DocFreqByTermCurrentQuery[['DocID', 'QueryDotProduct']].groupby('DocID').agg('sum')

                    # merge in document vector lengths to calc sim score
                    DVITableAggReset = DVITableAgg.reset_index()
                    QuerySimScore = pd.merge(QuerySimScore, DVITableAggReset[['DocID', 'DocVectorLength']], on='DocID')
                    QuerySimScore['CosineSimScore'] = QuerySimScore['QueryDotProduct'] / (QuerySimScore['DocVectorLength'] * currentQueryVectorLength)

                    # sort and find top 1000 records, reset index
                    QuerySimScore = QuerySimScore.sort_values(by='CosineSimScore', ascending=False)
                    newRankedList = QuerySimScore.head(1000)
                    newRankedList = newRankedList.reset_index()

                    # complete our new records for storage
                    newRankedList['QueryID'] = [currentDocID] * len(newRankedList)
                    newRankedList['Q0'] = ['Q0'] * len(newRankedList)
                    newRankedList['Rank'] = newRankedList.reset_index().index
                    newRankedList['JHID'] = ['tbaird7'] * len(newRankedList)

                    # round and adjust for NaN in sim score
                    newRankedList = newRankedList.round({'CosineSimScore': 6})
                    newRankedList['CosineSimScore'] = newRankedList['CosineSimScore'].fillna(0)

                    # only output columns needed
                    newRankedList = newRankedList[['QueryID', 'Q0', 'DocID', 'Rank', 'CosineSimScore', 'JHID']]

                    # add new values to our output table
                    outputRankedList = pd.concat([outputRankedList, newRankedList], ignore_index=True)

                    print(outputRankedList)
                # process another line of text
                else:
                    nextLine = [word.lower() for word in nextLine]
                    nextLine = [word.strip() for word in nextLine]
                    nextLine = [word.strip(string.punctuation) for word in nextLine]
                    # nextLine = [ps.stem(word) for word in nextLine]
                    currentParagraph.extend(nextLine)

    print(f"*{datetime.datetime.now()}* Done Doc Processing")

    # Print to text file
    outputRankedList.to_csv('tbaird7.txt', sep=' ', header=True, index=False)

    endTime = time.time()
    elapsedTime = endTime - startTime
    print(f"Time to complete: {elapsedTime}")
