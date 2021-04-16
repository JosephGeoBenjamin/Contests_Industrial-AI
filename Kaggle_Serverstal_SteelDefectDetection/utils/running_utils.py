import csv
import torch
from torchsummary import summary

def LOG2CSV(data, csv_file, flag = 'a'):
    '''
    data: List of elements to be written
    '''
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()




def TEST_MODEL(model, inp_size=(10,10,10)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=inp_size)