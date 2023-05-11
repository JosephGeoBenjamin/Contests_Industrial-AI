'''
'''
import sys, os
from tqdm.auto import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader

import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.dataUtils import SeqNESPData, AminoStrawboss
from utilities.metricUtils import AccuracyComputer
from utilities.inferUtils import kaggle_submission
from algorithms.rnn import RNNEncoder
from algorithms.regress import MLPRegress

##===== Init Setup =============================================================
rutl.START_SEED()

INST_NAME = "Train_112"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

##------------------------------------------------------------------------------

LOG_PATH = "hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"check"
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")


##===== Running Configuration =================================================

num_epochs = 500
batch_size = 128
accum_grad = 1
workers = 24
learning_rate = 1e-3
pretrain_wgt_path = None
amino_obj = AminoStrawboss()

train_dataset = SeqNESPData(csv_file='datasets/train_split.csv',
                        src_amino_obj = amino_obj,
                        padding = True, max_seq_size = 2200)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=workers)

val_dataset = SeqNESPData(csv_file='datasets/valid_split.csv',
                        src_amino_obj = amino_obj,
                        padding = True)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=workers)

print("Train Count:>>", len(train_dataset))
print("Validation Count:>>", len(val_dataset))

##===== Model Configuration =================================================

input_dim = amino_obj.size()
enc_emb_dim = 1024
enc_hidden_dim = 1024
enc_layers = 2
enc_bidirect = True
fc_layers = 3
dropout_p = 0.3
class TheModel(torch.nn.Module):
    def __init__(self):
        super(TheModel, self).__init__()

        self.encode = RNNEncoder(
                        input_dim = input_dim, embed_dim= enc_emb_dim,
                        hidden_dim= enc_hidden_dim, rnn_type = 'lstm',
                        enc_layers = enc_layers, bidirectional = enc_bidirect,
                        dropout = 0,)
        self.regress = MLPRegress(
                        input_dim= enc_hidden_dim*enc_layers*(1+enc_bidirect),
                        out_dim=1, layers=fc_layers, dropout=dropout_p)

    def forward(self, x, x_sz):
        xout = self.encode(x, x_sz)
        out = self.regress(xout)
        return out


model = TheModel().to(device)

## ----- Load Pretrain ---------------------------------------------------------

# hi_emb_vecs = np.load("data/embeds/hi_char_512_ftxt.npy")
# model.decoder.embedding.weight.data.copy_(torch.from_numpy(hi_emb_vecs))

# en_emb_vecs = np.load("data/embeds/en_char_512_ftxt.npy")
# model.encoder.embedding.weight.data.copy_(torch.from_numpy(en_emb_vecs))

# model = rutl.load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified

##------ Model Details ---------------------------------------------------------
rutl.count_train_param(model)
print(model)


##====== Optimizer Zone ========================================================


criterion = torch.nn.L1Loss()
# criterion = torch.nn.MSELoss()
# class_weight = torch.from_numpy(train_dataset.tgt_class_weights).to(device)  )

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
    for ith, (src, tgt, src_sz) in enumerate(tqdm(train_dataloader)):

        src = src.to(device)
        tgt = tgt.to(device)

        #--- forward ------
        output = model(x = src, x_sz =src_sz)
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
    for jth, (v_src, v_tgt, v_src_sz) in enumerate(tqdm(val_dataloader)):
        v_src = v_src.to(device)
        v_tgt = v_tgt.to(device)
        with torch.no_grad():
            v_output = model(x = v_src, x_sz = v_src_sz)
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
    test_dataset = SeqNESPData( csv_file='datasets/test.csv',  mode='infer',
                            src_amino_obj = amino_obj,  padding = True)

    test_dataloader = DataLoader(test_dataset, batch_size=infer_batch,
                                shuffle=False, num_workers=0)

    ### Model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    ### Run
    preds = []
    for data, name, data_sz in test_dataloader:
        #name: seq_id [Int]
        outval = model(data.to(device), x_sz = data_sz)
        for n, c in zip(name.detach().cpu().numpy(),
                        outval.detach().cpu().numpy()):
            preds.append([n,c[0]])

    ### Save
    kaggle_submission(preds, filename=model_path[:-4])

if __name__ =="__main__":

    training_procedure()
    # inference_procedure('hypotheses/Train_101/check_model.pth')