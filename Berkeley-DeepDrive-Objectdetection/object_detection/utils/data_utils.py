import json
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import Dataset
from torchvision import transforms


class BerkeleyObjectDetect2dData(Dataset):
    '''
    Berkeley Deep Drive dataset
    json_file - format:
    list({ "name": "image_id.jpg",
            "labels": [{
                "category": "object_class",
                # relative Positions
                "box2d": { "x1": float, "y1": float, "x2": float, "y2": float}
            },]
            "attributes": { "weather": "overcast", "scene": "city street", "timeofday": "daytime"},
        })
    '''
    def __init__(self, json_file, image_folder, category_list,
                divisibility = 1, #int
                scale_factor = 1, # float
                fix_img_size = None, # tuple-> (h_pix, w_pix)
                ):
        """
        divisibility: resize the image to upper multiple,
                        to prevent issues in upsample and downsample layers
        scale_factor: rescale image size,
                        to reduce GMACs (training time) on larger images, also for different scale training
        fix_img_size: Common standard image size,
                        to which all the image must be resized for stable training
        """

        self.json_data = json.load(open(json_file))
        self.image_folder = image_folder
        self.scale_factor = scale_factor
        self.divisibility = divisibility
        self.fix_img_size = fix_img_size

        self.catg2id = {}
        for i, c in enumerate(category_list):
            self.catg2id[c] = i

        # self.target_struct={}
        # for i in ['categ_id', 'cx', 'cy', 'h', 'w', 'x1', 'y1', 'x2', 'y2']:
        #     self.target_struct[i] = torch.zeros((self.max_num_object))

        if fix_img_size:
            print("Standard Image size set to ", (fix_img_size[0],fix_img_size[1]) )

    def __getitem__(self, idx):
        '''
        All sizes will be converted relative notation
        Image will be resized to multiple of reduction size, to support upscaling
        '''

        # img :shape: width, height
        img = Image.open(os.path.join(self.image_folder, self.json_data[idx]["name"]) )
        img = img.convert("RGB")
        org_img_size = {"h":img.size[1], "w":img.size[0]}

        if self.divisibility > 1 or self.scale_factor != 1 or self.fix_img_size:
            if self.fix_img_size:
                h,w = self.fix_img_size
            else:
                h,w = img.size
            h,w = h*self.scale_factor, w*self.scale_factor
            h = np.ceil(h / self.divisibility)* self.divisibility
            w = np.ceil(w / self.divisibility)* self.divisibility
            img = img.resize( (int(h),int(w)) )

        # img_tnsr :shape:channel, height, width
        img_tnsr = transforms.ToTensor()(img)

        target=[]
        for data in self.json_data[idx]["labels"]:
            tgt = {}
            tgt["categ_id"] = self.catg2id[data["category"]]
            tgt["category"] = data["category"]
            tgt["cx"] = (data["box2d"]["x1"] + data["box2d"]["x2"]) / (2*org_img_size['w'])
            tgt["cy"] = (data["box2d"]["y1"] + data["box2d"]["y2"]) / (2*org_img_size['h'])
            tgt["w"] = abs(data["box2d"]["x2"] - data["box2d"]["x1"]) / org_img_size['w']
            tgt["h"] = abs(data["box2d"]["y2"] - data["box2d"]["y1"]) / org_img_size['h']
            tgt['x1'] = data["box2d"]["x1"] / org_img_size['w']
            tgt['y1'] = data["box2d"]["y1"] / org_img_size['h']
            tgt['x2'] = data["box2d"]["x2"] / org_img_size['w']
            tgt['y2'] = data["box2d"]["y2"] / org_img_size['h']
            target.append(tgt)

        return img_tnsr, target

    def __len__(self):
        return len(self.json_data)

    def data_collate(self, batch):
        data = [item[0] for item in batch]
        max_w = max([d.shape[-1] for d in data])
        max_h = max([d.shape[-2] for d in data])

        pad_data = []
        for d in data:
            pd = nnf.pad( input=d, pad= (0, max_w - d.shape[-1], 0, max_h - d.shape[-2]),
                      mode='constant', value=0)
            pad_data.append(pd)
        pad_tnsr = torch.stack(pad_data, dim=0)

        target = [item[1] for item in batch]
        return (pad_tnsr, target)




