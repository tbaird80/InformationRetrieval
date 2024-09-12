# Press the green button in the gutter to run the script.
import re
import pandas as pd
import sys

def findIndex(rawValue):
    # knowing standard format for ID, we can remove the irrelevant characters
    valuesToRemove = ['id=', '>']
    indexValue = rawValue
    for currentRemoveValue in valuesToRemove:
        indexValue = indexValue.replace(currentRemoveValue, '')

    # return the remaining value as an integer
    return int(indexValue)


def removeUselessChar(rawValue):
    # below are characters that we want to remove from strings
    newValue = rawValue
    # these values we want to remove with empty string
    valuesToRemove = ['"', "'", "`", "\n", "â€™", "{", "}", "[", "]", "#", "$", "’", ".", ";", "”"]
    # these we want to remove but replace with a blank space
    valuesToSpaceReplace = ["\t", "â€”", "&", "~"]

    # iterate through both lists to update our value
    for currentRemoveValue in valuesToRemove:
        newValue = newValue.replace(currentRemoveValue, '')
    for currentNewSpace in valuesToSpaceReplace:
        newValue = newValue.replace(currentNewSpace, ' ')

    return newValue


def addValues(currentTable, newValues, currentDocID):
    # create new row for our output dataframe
    newRows = pd.DataFrame({
        'DocID': [currentDocID] * len(newValues),  # Repeat the name for each city
        'Value': newValues
    })

    # add to existing dataframe and return the updated table
    currentTable = pd.concat([currentTable, newRows], ignore_index=True)
    return currentTable


if __name__ == '__main__':
    # init our variables needed to track reading of
    fullLog = pd.DataFrame()
    currentID = 0

    # read in the file name which is passed as a command line arg
    currentFileName = sys.argv[1]
    print(f"Reading in text file: {currentFileName}")

    # create our output file to store our results
    outputFileName = currentFileName.replace(".txt", "Output.txt")
    outputFile = open(outputFileName, "w")
    # write opening line of document
    outputFile.write(f"This is our output for {currentFileName}\n\n")

    # iterate through the open file line by line
    with open(currentFileName, 'r', encoding='utf-8') as fullFile:
        for nextLine in fullFile:
            # remove characters that we do not need
            cleanLine = removeUselessChar(nextLine)
            # empty lines and end of paragraphs we can skip
            if cleanLine == '' or cleanLine == '</P>':
                pass
            else:
                # set all to lower case
                cleanLine = cleanLine.lower()
                # these values we want to split into new word as they are natural breaks in a sentence or a paragraph
                cleanLine = re.split(r'[—_|:/\s,()-]', cleanLine)
                # check if beginning of new paragraph where we will extract our paragraph id
                if cleanLine[0] == '<p':
                    newIndex = findIndex(cleanLine[1])
                    currentID = newIndex
                    print(f"Processing document {newIndex}")
                # otherwise it is a good line to process for our tokens
                else:
                    fullLog = addValues(fullLog, cleanLine, currentID)

    # remove any stray empty values
    fullLog = fullLog[fullLog['Value'] != '']

    # find the summary of number of paragraphs, unique words, and total words
    numParagraphs = fullLog['DocID'].nunique()
    numUniqueWords = fullLog['Value'].nunique()
    numTotalWords = len(fullLog)

    # write our initial summary items
    outputFile.write(f"We have processed {numParagraphs} total paragraphs\n")
    outputFile.write(f"We have found a vocabulary size of {numUniqueWords}\n")
    outputFile.write(f"We have found a collection size of {numTotalWords}\n\n")

    # group the words by collection frequency
    collectionFreq = fullLog.groupby('Value').agg({'DocID': 'count'})
    collectionFreq = collectionFreq.sort_values(by='DocID', ascending=False)
    # rename the count column
    collectionFreq = collectionFreq.rename(columns={'DocID': 'ColFreq'})

    # group the words by document frequency
    # first drop duplicates that represent multiple instances of word in document
    documentFreq = fullLog.drop_duplicates()
    # then find count of each value which now represents the number of documents it appears in
    documentFreq = documentFreq.groupby('Value').agg({'DocID': 'count'})
    documentFreq = documentFreq.sort_values(by='DocID', ascending=False)
    # rename the count column
    documentFreq = documentFreq.rename(columns={'DocID': 'DocFreq'})

    # join together the collection and document freq
    collectionFreq = collectionFreq.join(documentFreq).reset_index()

    # print out information of top 50 records
    for currentRank in range(100):
        currentRankID = currentRank + 1
        currentWord = collectionFreq.loc[currentRank, 'Value']
        currentColFreq = collectionFreq.loc[currentRank, 'ColFreq']
        currentDocFreq = collectionFreq.loc[currentRank, 'DocFreq']

        outputFile.write(f"{currentRankID}: '{currentWord}' has a collection frequency of {currentColFreq} and document frequency of {currentDocFreq}\n")

    # print info for words 500, 1000, and 1500
    otherIndices = [500, 1000, 5000]
    for currentRankID in otherIndices:
        currentRank = currentRankID - 1
        currentWord = collectionFreq.loc[currentRank, 'Value']
        currentColFreq = collectionFreq.loc[currentRank, 'ColFreq']
        currentDocFreq = collectionFreq.loc[currentRank, 'DocFreq']

        outputFile.write(f"{currentRankID}: '{currentWord}' has a collection frequency of {currentColFreq} and document frequency of {currentDocFreq}\n")

    # find words only in one document
    singleDocWords = documentFreq[documentFreq['DocFreq'] == 1].reset_index()
    numSingleDocWords = len(singleDocWords)
    percentSingleDocWord = round((numSingleDocWords / numUniqueWords) * 100, 3)
    outputFile.write(f"\nThe number of words that only appeared in one document are: {numSingleDocWords}")
    outputFile.write(f"\nThe percentage of these terms relative to the full dictionary is: {percentSingleDocWord}%")
