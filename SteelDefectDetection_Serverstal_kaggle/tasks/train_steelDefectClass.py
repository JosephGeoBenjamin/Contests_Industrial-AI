''' Binary Classifier

Output: Binary Classifier for 4+1 classes seperately

'''
import os
from tqdm import tqdm

import torch
from torchvision import models as tvmodels
from torch.utils.data import DataLoader

from utils.severstalData_utils import SeverstalClassifierData
import utils.metrics_utils as metrics
from utils.running_utils import LOG2CSV, TEST_MODEL

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##===== Init Setup =============================================================
INST_NAME = "test_class"

num_epochs = 10000
batch_size = 3
acc_batch = 1
learning_rate = 1e-5

##------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_PATH = "hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"/weights/"
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")

##===== Datasets ===============================================================

DATASET_PATH='datasets/severstal-steel-defect-detection/'

train_dataset = SeverstalClassifierData(csv_file= DATASET_PATH +'steel_train.csv',
                                    root_dir= DATASET_PATH +'/train_images',
                                    device = device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True)

valid_dataset = SeverstalClassifierData(csv_file= DATASET_PATH+'/steel_valid.csv',
                                    root_dir= DATASET_PATH +'/train_images',
                                    device = device)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=True)

# print(valid_dataset.__getitem__(10))
# for x in valid_dataloader:
#     print(x)
#     break

##===== Model Configuration ====================================================

model = tvmodels.resnet.ResNet(block= tvmodels.resnet.Bottleneck,
                        layers=[3, 4, 6, 3], # renet50
                        num_classes=5 )

TEST_MODEL(model, (3, 1600, 256))

##====== Optimizer Zone ========================================================

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

criterion = torch.nn.BCEWithLogitsLoss()
def loss_estimator(output, target):
    loss = criterion(output, target)
    return loss

acc_est = metrics.ClassifierAccuracyComputer(5)

##=========  MAIN  ===========================================================

if __name__ =="__main__":

    best_loss = float("inf")
    best_acc = 0
    for epoch in range(num_epochs):
        #--- Train
        model.train()
        acc_loss = 0
        running_loss = []
        for ith, (img, gt) in enumerate(train_dataloader):
            img = img.to(device)
            gt = gt.to(device)

            #--- forward
            output = model(img)

            loss = loss_estimator(output, gt) / acc_batch
            acc_loss += loss

            #--- backward
            loss.backward()
            if ( (ith+1) % acc_batch == 0):
                optimizer.step()
                optimizer.zero_grad()
                print('epoch[{}/{}], Mini Batch-{} loss:{:.4f}'
                    .format(epoch+1, num_epochs, (ith+1)//acc_batch, acc_loss.data))
                running_loss.append(acc_loss.data)
                acc_loss=0
                # break
        LOG2CSV(running_loss, LOG_PATH+"/trainLoss.csv")

        #--- Validate
        model.eval()
        val_loss = 0
        acc_est.reset()
        for jth, (val_img, val_gt) in enumerate(tqdm(valid_dataloader)):
            val_img = val_img.to(device)
            val_gt = val_gt.to(device)
            with torch.no_grad():
                val_output = model(val_img)
                val_loss += loss_estimator(val_output, val_gt)
                acc_est.update_accuracy(val_output, val_gt)
            # break
        val_loss = val_loss / len(valid_dataloader)
        val_acc, val_acc_list = acc_est.get_acc()

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{}'
              .format(epoch+1, num_epochs, val_loss.data, val_acc))
        LOG2CSV([val_loss.item(), val_acc] + val_acc_list,
                    LOG_PATH+"/validLoss.csv")

        #--- save Checkpoint
        if val_loss < best_loss:
            print("***saving best optimal state [Loss:{}] ***".format(val_loss.data))
            best_loss = val_loss
            best_acc = val_acc
            torch.save(model, WGT_PREFIX+"model.pth")
            LOG2CSV([epoch+1,val_loss.item(), val_acc] + val_acc_list,
                    LOG_PATH+"/bestCheckpoint.csv")
