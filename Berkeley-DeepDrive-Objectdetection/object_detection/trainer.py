'''
'''
import numpy as np
import os
import sys
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
# torch.autograd.set_detect_anomaly(True)

from utils.data_utils import BerkeleyObjectDetect2dData
from utils.loss_utils import YoloV3Loss
from utils.accuracy_utils import ObjectDetectAccuracy
import utils.misc_utils as mutl
from yolo_v3 import YoloV3, YoloV3Inference

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True

##===== Init Setup =============================================================

INST_NAME = "Training_x"

LOG_PATH = "object_detection/hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"weights/"+INST_NAME
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


##===== Data Configuration =================================================
object_category = [ "bike", "bus", "car", "motor", "person",
"rider", "traffic light", "traffic sign", "train", "truck"  ]
anchors = [
        [0.29010481, 0.43303778],
        [0.15700936, 0.21523859],
        [0.01428714, 0.02611827],
        [0.03419133, 0.06047727],
        [0.07893973, 0.11255236],    ] # Relative

num_epochs = 1000
batch_size = 1
acc_grad = 1
learning_rate = 1e-5
pretrain_wgt_path = None


divisibility = 32 # based on darknet-upscaling
scale_factor = 0.25

train_dataset = BerkeleyObjectDetect2dData(
    json_file="data/object_detect/object_detect_2dbbox_train.json",
    image_folder="data/BDD_100k/train/",
    category_list= object_category,
    divisibility=divisibility,
    scale_factor=scale_factor,
    fix_img_size=None, # (416, 416)
)

# val_count = int(len(dataset) * 0.2  )
# train_count = len(dataset) - val_count

# train_dataset, val_dataset = random_split( dataset,
#             lengths = [ train_count, val_count ] )

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0,
                                collate_fn = train_dataset.data_collate,
                                )

val_dataset = BerkeleyObjectDetect2dData(
    json_file="data/object_detect/object_detect_2dbbox_val.json",
    image_folder="data/BDD_100k/val/",
    category_list= object_category,
    divisibility=divisibility,
    scale_factor=scale_factor,
)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0,
                                collate_fn = train_dataset.data_collate,
                                )

# for i in range(len(train_dataset)):
#     print(train_dataset.__getitem__(i))

# for batch in train_dataloader:
#     print(batch)

##===== Model Configuration ====================================================

model = YoloV3("darknet53", num_class=len(object_category),
                            num_anchor=len(anchors),)
model = model.to(device)

model_inference = YoloV3Inference(model_fwd = model.forward,
                                    anchors = anchors,
                                    classes = object_category,
                                    conf_thresh = 0.6,
                                    nms_thresh = 0.5,
                                    device = device, )

##------ Model Details ---------------------------------------------------------
mutl.count_train_param(model)
# print(model)

##====== Optimizer Zone ========================================================
criterion = YoloV3Loss( classes=object_category,
    anchors = anchors,
    scale_anchor=1,
    iou_thresh=0.3,
    device = device,
    )
loss_estimator = criterion.compute_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=0.01)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

accuracy_scorer = ObjectDetectAccuracy(iou_thresh = 0.7)


#===============================================================================

if __name__ =="__main__":

    best_loss = float("inf")
    best_accuracy = 0
    for epoch in range(num_epochs):

        #-------- Training -------------------
        model.train()
        acc_loss = 0
        running_loss = []

        for ith, (src_img, label) in enumerate(train_dataloader):
            src_img = src_img.to(device)

            #--- forward ------
            output = model(src_img)
            loss = loss_estimator(output, label) / (acc_grad * batch_size)
            acc_loss += loss

            #--- backward ------
            loss.backward()
            if ( (ith+1) % acc_grad == 0):
                optimizer.step()
                optimizer.zero_grad()

                print('epoch[{}/{}], MiniBatch-{} loss:{:.4f}'
                    .format(epoch+1, num_epochs, (ith+1)//acc_grad, acc_loss.data))
                running_loss.append(acc_loss.item())
                acc_loss=0
                # break

        mutl.LOG2CSV(running_loss, LOG_PATH+"trainLoss.csv")

        ###------- Validate ---------------------
        if (epoch%5 == 0):
            model.eval()
            val_loss = 0
            val_accuracy = [None] * 3
            for jth, (v_src, v_label) in enumerate(tqdm(val_dataloader)):
                v_src = v_src.to(device)

                with torch.no_grad():
                    v_out_tnsr, v_out_dict = model_inference.run(v_src)
                    val_loss += loss_estimator(v_out_tnsr, v_label)
                    accuracy_scorer.update(v_label, v_out_dict)
                #break
            val_loss = val_loss / len(val_dataloader)
            val_accuracy = accuracy_scorer.get( divideby = len(val_dataloader) )
            accuracy_scorer.reset()

            print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{:.4f}'
                .format(epoch+1, num_epochs, val_loss.data, val_accuracy))
            mutl.LOG2CSV([val_loss.item(), *val_accuracy ],
                        LOG_PATH+"valLoss.csv")

            #-------- save Checkpoint -------------------
            # if val_accuracy > best_accuracy:
            if val_loss < best_loss:
                print("***saving best optimal state [Loss:{} Accur:{}] ***".format(val_loss.data,val_accuracy.data) )
                best_loss = val_loss
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), WGT_PREFIX+"_model-{}.pth".format(epoch+1))
                mutl.LOG2CSV([epoch+1, val_loss.item(), val_accuracy.item()],
                        LOG_PATH+"bestCheckpoint.csv")
        ###------- End Validation --------------------

        # LR step
        # scheduler.step()