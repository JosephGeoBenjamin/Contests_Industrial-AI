import torch
import csv
import os

def load_pretrained(model, weight_path, flexible = False):
    if not weight_path:
        return model

    pretrain_dict = torch.load(weight_path)
    model_dict = model.state_dict()
    if flexible:
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    print("Pretrained layers:", pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model

def count_train_param(model):
    train_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(train_params_count))
    return train_params_count

def freeze_params(model, exclusion_list = []):
    ## TODO: Exclusion lists
    for param in model.parameters():
        param.requires_grad = False
    return model


def LOG2CSV(data, csv_file, flag = 'a'):
    '''
    data: List of elements to be written
    '''
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()