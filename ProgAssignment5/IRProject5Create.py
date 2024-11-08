import datetime
import time
import csv

if __name__ == '__main__':

    startTime = time.time()

    # Open and read the .tsv file
    with open("greek.tsv", mode='r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            # print(row)  # Process each row as needed
            pass

    endTime = time.time()

    elapsedTime = endTime - startTime

    print(f"*{datetime.datetime.now()}* Done Creation Program")
    print(f"Time to complete: {elapsedTime}")

