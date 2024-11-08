import pandas as pd
import datetime
import struct
import numpy as np

if __name__ == '__main__':
    print(f"*{datetime.datetime.now()}* Starting Document Vector Length Processing")

    # our file names
    dictFile = "animalDict.csv"
    invertedFilePath = "animalInvertedFile"

    # read dictionary from file and find the total size of the inverted file using last item on list
    fullFreqTable = pd.read_csv(dictFile)

    # open binary file
    with open(invertedFilePath, 'rb') as invertedFile:
        # start at the beginning of the file
        inverseFileContent = invertedFile.read()

    # convert the binary file into list of integers
    inverseFileContent = struct.unpack(f'{len(inverseFileContent)//4}i', inverseFileContent)

    # create a dataframe out of the list, evens are doc ids and odds are doc freq
    docIDs = inverseFileContent[::2]
    docTermFreqs = inverseFileContent[1::2]

    # create parallel list of terms that match the number of postings for each
    termList = fullFreqTable['Term'].to_list()
    termDFs = fullFreqTable['DocFreq'].to_list()
    termListExpanded = [value for value, freq in zip(termList, termDFs) for _ in range(freq)]

    # create our table with our values, merge in our IDF
    DVITable = pd.DataFrame({'Term': termListExpanded, 'DocID': docIDs, 'DocTermFreq': docTermFreqs})
    DVITable = pd.merge(DVITable, fullFreqTable[['Term', 'IDF']], on='Term')

    # calc our weight
    DVITable['DocVectorLength'] = (DVITable['DocTermFreq'] * DVITable['IDF'])**2
    DVITableAgg = DVITable[['DocID', 'DocVectorLength']].groupby('DocID').agg('sum')
    DVITableAgg['DocVectorLength'] = np.sqrt(DVITableAgg['DocVectorLength'])

    print(f"*{datetime.datetime.now()}* Done Document Vector Length Processing")
