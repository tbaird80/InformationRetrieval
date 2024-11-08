import string
import math
import pandas as pd
import sys
import datetime
import io
import struct
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

if __name__ == '__main__':

    # our file names
    dictFile = "totDict.csv"
    invertedFile = "totInvertedFile"
    docVectorLengthFile = 'totDocVectorLength.csv'

    # read dictionary from file and find the total size of the inverted file using last item on list
    fullFreqTable = pd.read_csv(dictFile, index_col='Term')
    docVectorLengthTable = pd.read_csv(docVectorLengthFile, index_col='DocID')

    # init our dictionaries to track our dictionary universes
    currentQueryTermsWeight = {}
    relevantDocumentScores = {}

    # our output table that we want to track as we go
    outputRankedList = pd.DataFrame(columns=['QueryID', 'Q0', 'DocID', 'Rank', 'SimScore', 'JHID'])

    # create our list of stop words an initialize our porter stemmer
    nltk.download('stopwords')
    stopwordsList = set(stopwords.words('english'))
    ps = PorterStemmer()

    # read in the file name which is passed as a command line arg
    currentFileName = "train.queries.txt"
    print(f"Reading in text file: {currentFileName}")

    print(f"*{datetime.datetime.now()}* Starting Query Processing")
    # iterate through the open file line by line
    with open(currentFileName, 'r', encoding='utf-8') as queryFile, open(invertedFile, 'rb') as invertedFile:
        currentQuery = []
        currentDocumentSet = {}
        for rawNextLine in queryFile:
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
                nextLine = nextLine.split()

                # pass if line empty
                if len(nextLine) == 0:
                    pass
                # update the document ID where relevant
                elif nextLine[0] in ('<P', '<Q'):
                    currentDocID = int(nextLine[2].rstrip('>'))
                    # print(f"*{datetime.datetime.now()}* Processing Doc {currentQueryID}")
                # hit end of paragraph, time to empty out the tokens
                elif rawNextLine in ('</P>\n', '</Q>\n'):
                    tempDict = {}
                    for currentToken in currentQuery:
                        # skip blanks, single characters and numbers
                        if currentToken == "" or len(currentToken) == 1 or currentToken.isnumeric():
                            pass
                        elif currentToken in tempDict:
                            tempDict[currentToken] += 1
                        else:
                            tempDict[currentToken] = 1

                    currentQueryVectorLength = 0
                    for token, count in tempDict.items():
                        if token in fullFreqTable.index:
                            # calculate the weight of the query term
                            currentTokenIDF = fullFreqTable.loc[token, 'IDF']
                            currentQueryTokenWeight = currentTokenIDF * count
                            currentQueryTermsWeight[token] = currentQueryTokenWeight
                            currentQueryVectorLength += currentQueryTermsWeight[token]**2

                            # grab the inverted file information
                            currentTokenOffset = fullFreqTable.loc[token, 'Offset']
                            currentTokenDocFreq = fullFreqTable.loc[token, 'DocFreq']

                            for currentDocInverted in range(currentTokenDocFreq):
                                # move to current offset and read next 8 bytes
                                invertedFile.seek(currentTokenOffset)
                                currentPosting = invertedFile.read(8)

                                # read 4 bytes each for doc id and doc freq
                                currentDocID, currentDocTermFreq = struct.unpack('ii', currentPosting)
                                currentDocTermWeightDotProduct = currentDocTermFreq * currentTokenIDF * currentQueryTokenWeight
                                if currentDocID in currentDocumentSet:
                                    currentDocumentSet[currentDocID] += currentDocTermWeightDotProduct
                                else:
                                    currentDocumentSet[currentDocID] = currentDocTermWeightDotProduct

                                # increment our offset by 8 to get to next postings list
                                currentTokenOffset += 8

                    # complete doc vector length calc of query by finding square root
                    currentQueryVectorLength = math.sqrt(currentQueryVectorLength)

                    # Create our dictionary table with all relevant information, drop document vector length when done
                    newRankedList = pd.DataFrame(list(currentDocumentSet.items()), columns=['DocID', 'DotProduct']).set_index('DocID')
                    newRankedList = newRankedList.join(docVectorLengthTable)
                    newRankedList['SimScore'] = newRankedList['DotProduct'] / (newRankedList['DocVectorLength'] * currentQueryVectorLength)
                    newRankedList = newRankedList.drop(columns=['DocVectorLength'])

                    # sort and find top 1000 records, reset index
                    newRankedList = newRankedList.sort_values(by='SimScore', ascending=False)
                    newRankedList = newRankedList.head(1000)
                    newRankedList = newRankedList.reset_index()

                    # complete our new records for storage
                    newRankedList['QueryID'] = [currentQueryID] * len(newRankedList)
                    newRankedList['Q0'] = ['Q0'] * len(newRankedList)
                    newRankedList['Rank'] = newRankedList.reset_index().index
                    newRankedList['JHID'] = ['tbaird7'] * len(newRankedList)

                    # add new values to our output table
                    outputRankedList = pd.concat([outputRankedList, newRankedList], ignore_index=True)

                # process another line of text
                else:
                    nextLine = [word.lower() for word in nextLine]
                    nextLine = [word.strip() for word in nextLine]
                    nextLine = [word.strip(string.punctuation) for word in nextLine]
                    # nextLine = [ps.stem(word) for word in nextLine if word not in stopwordsList]
                    currentQuery.extend(nextLine)

    # write our query ranked lists to file
    outputRankedListFile = 'totRankedList.csv'
    outputRankedList.to_csv(outputRankedListFile)

    print(f"*{datetime.datetime.now()}* Done Doc Processing")
