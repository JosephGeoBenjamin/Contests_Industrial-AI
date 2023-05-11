import csv
import os


def LOG2CSV(data, csv_file, flag = 'a'):
    '''
    data: List of elements to be written
    '''
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()


def LOG2TXT(data, txt_file, flag = 'a'):
    '''
    data: String to be written
    No formating here
    '''
    with open(txt_file, "a") as textFile:
        textFile.write(data)

