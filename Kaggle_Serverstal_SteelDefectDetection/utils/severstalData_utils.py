'''
Utilities for working with Severstal Steel defect dataset
Dataset: https://www.kaggle.com/c/severstal-steel-defect-detection/
'''
from os.path import join as OSPJ
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import pandas as pd
import imageio as imio
import csv

class SeverstalClassifierData(Dataset):
    ''' Used to load images and and truth
    '''
    def __init__(self, csv_file, root_dir, device='cpu'):
        self.device = device
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.num_dfct = 4+1

    def __getitem__(self, idx):
        """ Returns torch format CHW
        """

        img = self._load_image(idx)
        gt = self._load_class(idx)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        gt = torch.from_numpy(gt).type(torch.FloatTensor)

        return  Variable(img), Variable(gt)

    def __len__(self):
        return len(self.df)

    def _load_image(self, idx):
        img_path = OSPJ( self.root_dir , self.df.iloc[idx]['ImageId'])
        img = imio.imread(img_path)
        img = img.transpose(2,0,1)
        img = self.manual_transforms(img)
        return img

    def _load_class(self, idx):
        categ = self.df.iloc[idx]['Class']
        arr = np.zeros(self.num_dfct)
        r = [int(x) for x in str(categ)]
        arr[r] = 1
        return arr

    def manual_transforms(self, img):
        ''' Return CHW
        imput CHW
        '''
        control = np.random.randint(3)
        if control == 0: img = img[:,::-1,:].copy();  #H-flip
        elif control == 1: img = img[:,:,::-1].copy(); #V-flip
        elif control == 2:  img = img[:,::-1,::-1].copy(); #HV-flip

        return img


class SeverstalSteelData(Dataset):
    ''' Used to load images and numpy GroundTruth
    '''
    def __init__(self, img_dir, split_csv, rle_csv,
                        mode = 'defect_alone', #all
                        device='cpu'):
        self.device = device
        self.img_dir = img_dir
        df = pd.read_csv(split_csv)
        self.rle_df = pd.read_csv(rle_csv)

        if mode == 'defect_alone':
            self.img_list = list( pd.merge(df['ImageId'],
                                    self.rle_df['ImageId'],
                                    how='inner')['ImageId'] )
        else:
            self.img_list = df['ImageId']

    def __getitem__(self, idx):
        """ Returns torch format CHW
        """
        img_id = self.img_list[idx]
        img = imio.imread(OSPJ(self.img_dir,img_id))
        img = img.transpose(2,0,1) # 3,256,1600

        gt = self.compose_gt_mask(img_id)
        img, gt = self.manual_transforms(img, gt)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        gt = torch.from_numpy(gt).type(torch.FloatTensor)
        return img, gt

    def __len__(self):
        return len(self.img_list)


    def manual_transforms(self, img, target):
        ''' Return CHW
        imput CHW
        '''
        control = np.random.randint(3)
        if control == 0: img = img[:,::-1,:].copy(); target = target[:,::-1,:].copy(); #H-flip
        elif control == 1: img = img[:,:,::-1].copy(); target = target[:,:,::-1].copy(); #V-flip
        elif control == 2:  img = img[:,::-1,::-1].copy(); target = target[:,::-1,::-1].copy(); #HV-flip

        return img, target

    def rle_to_matrix(self, arr, cid, rle):
        rle = [ int(r) for r in rle.split() ]
        for i in range(0, len(rle), 2):
            lc = rle[i]; ln = rle[i+1]
            arr[cid-1, lc:lc+ln] = 1
        return arr

    def compose_gt_mask(self, img_id):
        rs = self.rle_df[self.rle_df['ImageId'] == img_id]
        new = np.zeros((4,1600*256))
        for j, r in rs.iterrows():
            new = self.rle_to_matrix(new, r['ClassId'], r['EncodedPixels'])
        ## change RLE to image structure
        new = new.reshape(4, 1600, 256)
        new = new.transpose(0,2,1)
        return new