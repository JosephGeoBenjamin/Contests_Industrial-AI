'''
Combination of BioEmbed + FullyConnected networks

Advised to precompute embeddings using tools/bioembed_generator.py to save time
during the training
'''
import sys, os
import h5py
import pandas as pd
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import AccuracyComputer
from utilities.inferUtils import kaggle_submission
from algorithms.rnn import RNNEncoder
from algorithms.regress import MLPRegress

##===== Init Setup =============================================================
rutl.START_SEED()

INST_NAME = "Train_FC2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

##------------------------------------------------------------------------------

LOG_PATH = "hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"check"
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")

##===== Running Configuration =================================================

num_epochs = 500
batch_size = 64
accum_grad = 1
workers = 4
learning_rate = 1e-4
pretrain_wgt_path = None


##======= Data Loader =================================================

class BioEmbedding_HDF5():
    def __init__(self, csv_file, mode = 'train',  # 'infer'
                    hdf5 = "",
                 ):
        #Load data
        if hdf5:
            self.h5_obj = h5py.File(hdf5)
        else:
            raise "Error HDF5 file needed"

        if mode == 'train':
            src_strs, tgt_vals, seq_ids = self._train_csvloader(csv_file)
            self.src = seq_ids # pass Id to fetch from hdf5
            self.tgt = tgt_vals
            self._getitem = self._train_item

        elif mode == 'infer':
            src_strs, seq_ids = self._test_csvloader(csv_file)
            self.src = seq_ids #pass Id to fetch from hdf5
            self.name = seq_ids
            self._getitem = self._test_item

        else:
            raise 'unknown task type'


    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return len(self.src)

    def _train_item(self, index):
        x = self.h5_obj[str(self.src[index])][:][0]
        y = self.tgt[index]
        return x, y

    def _test_item(self, index):
        x = self.h5_obj[str(self.src[index])][:][0]
        return x, self.src[index]

    def _train_csvloader(self, csv_file):
        ''' Convert CSV into required Data
        '''
        datadf = pd.read_csv(csv_file)
        x = []; y = []; w = []
        for i in range(len(datadf)):
            w.append(datadf.loc[i,'seq_id'])
            x.append(datadf.loc[i,'protein_sequence'])
            y.append(datadf.loc[i,'tm'])
        return x, y, w

    def _test_csvloader(self, csv_file):
        ''' Convert CSV into required Data
        '''
        datadf = pd.read_csv(csv_file)
        x = []; w = []
        for i in range(len(datadf)):
            w.append(datadf.loc[i, 'seq_id'])
            x.append(datadf.loc[i, 'protein_sequence'])
        return x, w


train_dataset = BioEmbedding_HDF5(csv_file='datasets/train_split.csv', mode="train",
                                  hdf5="datasets/Nesp-Train-bioembed.hdf5")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=workers)

val_dataset = BioEmbedding_HDF5(csv_file='datasets/valid_split.csv', mode="train",
                                  hdf5="datasets/Nesp-Valid-bioembed.hdf5")

val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=workers)


print(train_dataset.__getitem__(1))

##===== Model Configuration =================================================

fc_layers = 4
regress = MLPRegress( input_dim= 1024, out_dim=1,
                        layers=fc_layers, dropout=0.5)
model = regress.to(device)

##------ Model Details ---------------------------------------------------------
rutl.count_train_param(model)
print(model)


##====== Optimizer Zone ========================================================


criterion = torch.nn.L1Loss()
# criterion = torch.nn.MSELoss()

def loss_estimator(pred, truth):
    """
    pred: batch
    """
    truth = truth.type(torch.float)
    loss_ = criterion(pred.squeeze(), truth)
    return loss_  #torch.mean(loss_)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             weight_decay=0)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#===============================================================================

# Abstracted Routines # Involves Global objects instead of arguments

def train(epoch):
    model.train()
    accum_loss = 0
    running_loss = []
    accobj = AccuracyComputer()
    for ith, (src, tgt) in enumerate(tqdm(train_dataloader)):

        src = src.to(device)
        tgt = tgt.to(device)

        #--- forward ------
        output = model(x = src)
        loss = loss_estimator(output, tgt) / accum_grad
        accum_loss += loss
        accobj.add_entry(output, tgt)

        #--- backward ------
        loss.backward()
        if ( (ith+1) % accum_grad == 0):
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            running_loss.append(accum_loss.item())
            accum_loss=0
            # break

    train_accuracy = accobj.get_spearmanr()
    print('epoch[{}/{}], [===TRAIN===] loss:{:.4f} Accur:{:.4f}'.format(
                                            epoch+1, num_epochs,
                                            sum(running_loss)/len(running_loss),
                                            train_accuracy))
    lutl.LOG2CSV(running_loss, LOG_PATH+"trainLoss.csv")


def validate(epoch):
    model.eval()
    val_loss = 0
    v_accobj = AccuracyComputer()
    for jth, (v_src, v_tgt) in enumerate(tqdm(val_dataloader)):
        v_src = v_src.to(device)
        v_tgt = v_tgt.to(device)
        with torch.no_grad():
            v_output = model(x = v_src)
            val_loss += loss_estimator(v_output, v_tgt)
            v_accobj.add_entry(v_output, v_tgt)
        # break
    val_loss = val_loss / len(val_dataloader)
    val_accuracy = v_accobj.get_spearmanr()

    print('epoch[{}/{}], [---TEST---] loss:{:.4f}  Accur:{:.4f}'
            .format(epoch+1, num_epochs, val_loss.data, val_accuracy))
    lutl.LOG2TXT( 'epoch:{} :: TorchLoss:{} \n'.format(epoch, val_loss.data) + \
                    v_accobj.print_summary(ret=True) + '\n' ,
                    LOG_PATH+"valLoss.txt")
    return val_loss.data, val_accuracy



##---------------------------------------------------------

def training_procedure():
    best_loss = float("inf")
    best_accuracy = 0
    for epoch in range(num_epochs):
        train(epoch)
        v_loss, v_acc = validate(epoch)

        ### save Checkpoint  ###
        if v_acc > best_accuracy:
        # if v_loss < best_loss:
            print("***Saving best state*** [Loss:{:.4f} Accur:{:.4f}]".format(
                                                v_loss, v_acc) )
            best_loss = v_loss
            best_accuracy = v_acc
            torch.save(model.state_dict(), WGT_PREFIX+"_model.pth")
            lutl.LOG2CSV([epoch+1, v_loss.item(), v_acc], LOG_PATH+"bestCheckpoint.csv")


def inference_procedure(model_path):
    ### Data
    infer_batch = 8
    test_dataset = BioEmbedding_HDF5(csv_file='datasets/test.csv', mode="infer",
                                  hdf5="datasets/Nesp-Test-bioembed.hdf5")

    test_dataloader = DataLoader(test_dataset, batch_size=infer_batch,
                                shuffle=False, num_workers=0)

    ### Model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    ### Run
    preds = []
    for data, name, in tqdm(test_dataloader):
        #name: seq_id [Int]
        outval = model(data.to(device))
        for n, c in zip(name.detach().cpu().numpy(),
                        outval.detach().cpu().numpy()):
            preds.append([n,c[0]])

    ### Save
    kaggle_submission(preds, filename=model_path[:-4])

if __name__ =="__main__":

    # training_procedure()
    inference_procedure('hypotheses/Train_FC1/check_model.pth')