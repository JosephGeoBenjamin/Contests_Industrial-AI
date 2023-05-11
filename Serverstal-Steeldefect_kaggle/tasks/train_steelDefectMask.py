''' 4 class Mask Layers

Output: 4 binary mask layers each corresponding to a class

'''
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils.severstalData_utils import SeverstalSteelData
import utils.metrics_utils as metrics
from utils.running_utils import LOG2CSV, TEST_MODEL

import segmentation_models_pytorch as smp
# from networks.tiramisu import FCDenseNet57

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##===== Init Setup =============================================================
INST_NAME = "test_segm"

num_epochs = 10000
batch_size = 1
acc_batch = 1
learning_rate = 1e-5

##------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_PATH = "hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"/weights/"
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")

##===== Datasets ===============================================================

DATASET_PATH='datasets/severstal-steel-defect-detection/'

train_dataset = SeverstalSteelData( img_dir= DATASET_PATH+'/train_images',
                                    split_csv=DATASET_PATH+'/steel_train.csv',
                                    rle_csv=DATASET_PATH+'/train.csv',
                                    device = device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

test_dataset = SeverstalSteelData(img_dir= DATASET_PATH+'/train_images',
                                    split_csv=DATASET_PATH+'/steel_valid.csv',
                                    rle_csv=DATASET_PATH+'/train.csv',
                                    device = device)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

# print(train_dataset.__getitem__(2))

##===== Model Configuration ====================================================

model = smp.Linknet('se_resnext101_32x4d', classes=4,
                    activation=None, encoder_weights=None)
model = model.to(device)

TEST_MODEL(model, (3, 1600, 256))
##====== Optimizer Zone ========================================================

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

criterionDBCE = metrics.DiceBCELoss()
criterionFTversky = metrics.FocalTverskyLoss()
criterionFocal = metrics.FocalLoss()

def loss_estimator(output, target):

    lossDBCE = criterionDBCE(output, target)
#     lossFTversky = criterionFTversky(output, target)
#     lossFocal = criterionFocal(output, target)

    return lossDBCE


##=========  MAIN  ===========================================================

if __name__ =="__main__":

    best_loss = float("inf")
    for epoch in range(num_epochs):
        #------ Train -----------------------
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

        #------ Validate ---------------------
        model.eval()
        val_loss = 0
        val_accuracy = 0
        for jth, (val_img, val_gt) in enumerate(test_dataloader):
            val_img = val_img.to(device)
            val_gt = val_gt.to(device)
            with torch.no_grad():
                val_output = model(val_img)
                val_loss += loss_estimator(val_output, val_gt)

                val_accuracy += (1 - criterionDBCE(val_output, val_gt))
            # break
        val_loss = val_loss / len(test_dataloader)
        val_accuracy = val_accuracy / len(test_dataloader)

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data, val_accuracy.data))
        LOG2CSV([val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"/testLoss.csv")

        #--- save Checkpoint
        if val_loss < best_loss:
            print("***saving best optimal state [Loss:{}] ***".format(val_loss.data))
            best_loss = val_loss
            torch.save(model, WGT_PREFIX+"model.pth")
            LOG2CSV([epoch+1, val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"/bestCheckpoint.csv")
