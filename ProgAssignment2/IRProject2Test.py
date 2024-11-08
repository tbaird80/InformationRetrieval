import pandas as pd
import struct

if __name__ == '__main__':
    # our file names
    dictFile = "rfaDict.csv"
    invertedFile = "rfaInvertedFile"

    # all the token tests
    testOneList = ['crocodile', 'parrot', 'ethiopia', 'parthenon']
    testTwoList = ['hopkins', 'stanford', 'brown', 'college']

    # read dictionary from file
    fullFreqTable = pd.read_csv(dictFile, index_col='Term')

    # open binary file
    with open(invertedFile, 'rb') as f:

        print("****Test 1****")

        # iterate through all the tokens that make up the first test
        for currentTokenOne in testOneList:
            # initialize the output
            currentTokenPostingList = []

            # ensure that the token exists
            if currentTokenOne in fullFreqTable.index.tolist():
                # find the offset and doc freq to see where and how many postings stored
                currentTokenOffset = fullFreqTable.loc[currentTokenOne, 'Offset']
                currentTokenDocFreq = fullFreqTable.loc[currentTokenOne, 'DocFreq']

                # iterate through all postings
                for currentDoc in range(currentTokenDocFreq):
                    # move to current offset and read next 8 bytes
                    f.seek(currentTokenOffset)
                    currentPosting = f.read(8)

                    # read 4 bytes each for doc id and doc freq
                    currentDocID, currentDocFreq = struct.unpack('ii', currentPosting)
                    currentTokenPostingList.append((currentDocID, currentDocFreq))

                    # increment our offset by 8 to get to next postings list
                    currentTokenOffset += 8

            # print our result
            print(f"This is ({currentTokenOne}) posting list: {currentTokenPostingList}")

        print("\n****Test 2****")

        # iterate through our second list of tests
        for currentTokenTwo in testTwoList:
            # find the doc frequency in our dictionary
            currentTokenDocFreq = fullFreqTable.loc[currentTokenTwo, 'DocFreq']
            print(f"This is ({currentTokenTwo}) doc frequency: {currentTokenDocFreq}")

        print("\n****Test 3****")

        # initialize our output
        tonyDocIDs = []
        blairDocIDs = []

        # find the offset and doc freq to see where and how many postings stored
        tonyOffset = fullFreqTable.loc['tony', 'Offset']
        tonyDocFreq = fullFreqTable.loc['tony', 'DocFreq']

        # iterate through all tony postings
        for currentDoc in range(tonyDocFreq):
            # move to current offset and read next 4 bytes to get just the doc id
            f.seek(tonyOffset)
            currentPosting = f.read(4)
            currentDocID = struct.unpack('i', currentPosting)
            tonyDocIDs.extend(currentDocID)

            # increment our offset by 8 to get to next postings list
            tonyOffset += 8

        blairOffset = fullFreqTable.loc['blair', 'Offset']
        blairDocFreq = fullFreqTable.loc['blair', 'DocFreq']

        # iterate through all blair postings
        for currentDoc in range(blairDocFreq):
            # move to current offset and read next 4 bytes to get just the doc id
            f.seek(blairOffset)
            currentPosting = f.read(4)
            currentDocID = struct.unpack('i', currentPosting)
            blairDocIDs.extend(currentDocID)

            # increment our offset by 8 to get to next postings list
            blairOffset += 8

        # find the list intersection
        listIntersection = [commonDoc for commonDoc in blairDocIDs if commonDoc in tonyDocIDs]
        print(f"The words (tony) and (blair) occur together in documents: {listIntersection}")


