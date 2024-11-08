import string
import pandas as pd
import sys
import datetime
import io
import struct

if __name__ == '__main__':

    # init our dictionaries to track our dictionary universes
    docFreqCount = {}
    termFreqCount = {}
    postingsList = {}

    # read in the file name which is passed as a command line arg
    currentFileName = sys.argv[1]
    print(f"Reading in text file: {currentFileName}")

    print(f"*{datetime.datetime.now()}* Starting Doc Processing")
    # iterate through the open file line by line
    with open(currentFileName, 'r', encoding='utf-8') as fullFile:
        for nextLine in fullFile:
            if nextLine == '\n' or nextLine == '</P>\n':
                pass
            else:
                nextLine = nextLine.replace('—', ' ')
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

                # document 57510 is empty, so want to pass on that
                if len(nextLine) == 0:
                    pass
                elif nextLine[0] == '<P':
                    currentDocID = nextLine[1].lstrip('ID=')
                    currentDocID = int(currentDocID.rstrip('>'))
                    # print(f"*{datetime.datetime.now()}* Processing Doc {currentDocID}")
                else:
                    nextLine = [word.lower() for word in nextLine]
                    nextLine = [word.strip() for word in nextLine]
                    nextLine = [word.strip(string.punctuation) for word in nextLine]

                    tempDict = {}
                    for currentToken in nextLine:
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

    print(f"*{datetime.datetime.now()}* Done Doc Processing")

    # Create our dictionary table with all relevant information
    docFreqTable = pd.DataFrame(list(docFreqCount.items()), columns=['Term', 'DocFreq']).set_index('Term')
    termFreqTable = pd.DataFrame(list(termFreqCount.items()), columns=['Term', 'TermFreq']).set_index('Term')
    postingsListTable = pd.DataFrame(list(postingsList.items()), columns=['Term', 'PostingList']).set_index('Term')
    fullFreqTable = docFreqTable.join(termFreqTable)
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
    numberOfDocs = currentDocID
    vocabSize = len(fullFreqTable)
    collectionSize = fullFreqTable['TermFreq'].sum()
    print(f"Number of Documents: {numberOfDocs}")
    print(f"Vocab Size: {vocabSize}")
    print(f"Collection Size: {collectionSize}")

    # our file names for output
    dictFullFile = "rfaDictCheck.csv"
    dictFile = "rfaDict.csv"
    invertedFile = "rfaInvertedFile"

    # write our table to disk
    fullFreqTable.to_csv(dictFullFile, index=True)

    # drop the excess columns for the actual dictionary
    fullFreqTable.drop(['TermFreq', 'PostingList'], axis=1, inplace=True)
    fullFreqTable.to_csv(dictFile, index=True)

    # write binary file to disk
    binaryStream = io.BytesIO(buffer)
    with open(invertedFile, "wb") as f:
        f.write(binaryStream.read())

    print(f"*{datetime.datetime.now()}* Done Testing")
