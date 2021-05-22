'''
Submission file formatted according to Kaggle submission
'''

import os
import csv
import torch
import numpy as np
import imageio as imio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ROOT = 'datasets/severstal-steel-defect-detection/'
TEST_DIR = ROOT+'/test_images/'
img_list = sorted(os.listdir(TEST_DIR))

# Load Classifier
cpath = 'hypotheses/test_class/weights/model.pth'
cmodel = torch.load(cpath).to(device)
cmodel.eval()

def run_classifier(img):
    out = cmodel(img)
    out = torch.sigmoid(out)
    out = out.cpu().detach().numpy().squeeze(0)
    out = (out > 0.70).astype(int)
    if out[0]:
        res = []
    else:
        res = [ i for i,o in enumerate(out) if o]
    return res


# Load Segmenter
spath = 'hypotheses/test_segm/weights/model.pth'
smodel = torch.load(spath).to(device)
smodel.eval()

def run_segmenter(img):
    out = smodel(img)
    out = torch.sigmoid(out)
    out = out.cpu().detach().numpy().squeeze(0)
    out = (out > 0.65).astype(int)
    return out

##--------------------

def read_image(f):
    ipath = os.path.join(TEST_DIR, f)
    img = imio.imread(ipath)
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img).to(torch.float32)
    img = img.unsqueeze(0)
    return img

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_submission_rle(im_id, d_id, d_msk):
    res = [
        [im_id+"_1"], [im_id+"_2"],
        [im_id+"_3"], [im_id+"_4"],
    ]
    if not d_id: return res
    d_msk = d_msk.transpose(0,2,1)
    for j in d_id:
        res[j-1].append( rle_encode(d_msk[j-1]) )

    return res


if __name__=="__main__":

    kf = open(ROOT+'/submission.csv', 'w')
    ksub = csv.writer(kf)
    ksub.writerow(['ImageId_ClassId', 'EncodedPixels'])

    for i in img_list:
        img = read_image(i)
        d_id = run_classifier(img)
        if d_id: d_msk = run_segmenter(img)
        else:    d_msk = None

        res = get_submission_rle(i, d_id, d_msk)
        ksub.writerows(res)