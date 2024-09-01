# Press the green button in the gutter to run the script.
import re
import pandas as pd

def findIndex(rawValue):
    valuesToRemove = ['id=', '>']
    indexValue = rawValue
    for currentRemoveValue in valuesToRemove:
        indexValue = indexValue.replace(currentRemoveValue, '')

    return int(indexValue)

def removeUselessChar(rawValue):
    newValue = rawValue
    valuesToRemove = ['"', "'", "\n", "â€™"]
    valuesToSpaceReplace = ["\t", "â€”"]

    for currentRemoveValue in valuesToRemove:
        newValue = newValue.replace(currentRemoveValue, '')
    for currentNewSpace in valuesToSpaceReplace:
        newValue = newValue.replace(currentNewSpace, ' ')

    return newValue

def addValues(currentTable, newValues, currentDocID):
    # create new rows
    newRows = pd.DataFrame({
        'DocID': [currentDocID] * len(newValues),  # Repeat the name for each city
        'Value': newValues
    })

    # add to existing dataframe and return the updated table
    currentTable = pd.concat([currentTable, newRows], ignore_index=True)
    return currentTable


if __name__ == '__main__':
    currentFileName = 'test2.txt'
    fullLog = pd.DataFrame()
    currentID = 0

    with open(currentFileName, 'r') as fullFile:
        for nextLine in fullFile:
            cleanLine = removeUselessChar(nextLine)
            if cleanLine == '' or cleanLine == '</P>':
                pass
            else:
                cleanLine = cleanLine.lower()
                cleanLine = re.split(r'[._;|:/\s,()-]', cleanLine)
                if cleanLine[0] == '<p':
                    newIndex = findIndex(cleanLine[1])
                    currentID = newIndex
                    print(f"Processing document {newIndex}")
                else:
                    fullLog = addValues(fullLog, cleanLine, currentID)

    fullLog = fullLog[fullLog['Value'] != '']

    print(fullLog)
