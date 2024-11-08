import math
import pandas as pd
import sys
import datetime
import struct

if __name__ == '__main__':
    print(f"*{datetime.datetime.now()}* Starting Document Vector Length Processing")

    # our file names
    dictFile = "totDict.csv"
    invertedFilePath = "totInvertedFile"

    # read dictionary from file and find the total size of the inverted file using last item on list
    fullFreqTable = pd.read_csv(dictFile, index_col='Term')
    termList = fullFreqTable.index.to_list()
    lastTokenOffset = fullFreqTable.iloc[-1]['Offset']
    lastTokenDocFreq = fullFreqTable.iloc[-1]['DocFreq']
    totalDocumentRecords = int((lastTokenOffset + lastTokenDocFreq)//4)

    # our document vector length record
    docVectorLength = {}

    # open binary file
    with open(invertedFilePath, 'rb') as invertedFile:
        # start at the beginning of the file
        inverseFileContent = invertedFile.read()

    # convert the binary file into list of integers
    inverseFileContent = struct.unpack(f'{len(inverseFileContent)//4}i', inverseFileContent)

    for currentTerm in termList:
        print(currentTerm)
        currentInverseIndexLocation = int(fullFreqTable.loc[currentTerm, 'Offset']/8)
        currentTermDocFreq = int(fullFreqTable.loc[currentTerm, 'DocFreq'])
        currentTermIDF = fullFreqTable.loc[currentTerm, 'IDF']
        for currentRecordIndex in range(currentTermDocFreq):
            # read the next index values
            currentDocID, currentDocFreq = inverseFileContent[currentInverseIndexLocation], inverseFileContent[currentInverseIndexLocation+1]
            currentDocFreqSquare = (currentDocFreq * currentTermIDF) ** 2

            if currentDocID in docVectorLength:
                docVectorLength[currentDocID] += currentDocFreqSquare
            else:
                docVectorLength[currentDocID] = currentDocFreqSquare

            # increment our offset by 2 to get to next postings list
            currentInverseIndexLocation += 2

    # once all documents processed, we can square the values the get a final document vector length
    docVectorLength = {key: math.sqrt(value) for key, value in docVectorLength.items()}

    # create our document vector lengths
    docFreqTable = pd.DataFrame(list(docVectorLength.items()), columns=['DocID', 'DocVectorLength']).set_index('DocID')

    # write our document vector lengths to file
    docVectorLengthFile = 'totDocVectorLength.csv'
    docFreqTable.to_csv(docVectorLengthFile, index=True)

    print(f"*{datetime.datetime.now()}* Done Document Vector Length Processing")
