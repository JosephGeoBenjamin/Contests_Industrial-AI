'''
Reference:
1. https://pjreddie.com/media/files/papers/YOLOv3.pdf
2. https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e

'''

import torch
from torchvision.ops import nms
import torch.nn as nn
import numpy as np
from collections import OrderedDict

## 40584928 params

class DNBlock(nn.Module):
    def __init__(self, channels, res_unit):
        super(DNBlock, self).__init__()

        #  downsample
        self.down_sample = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1),     )

        self.res_layers = nn.ModuleDict() #OrderedDict
        for i in range(0, res_unit):
            self.res_layers["residual_{}".format(i)] = nn.Sequential(
                nn.Conv2d(channels[1], channels[0], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels[1]),
                nn.LeakyReLU(0.1),      )

    def forward(self, x):

        x_ = self.down_sample(x)
        residual = x_

        for res in self.res_layers:
            x_ = self.res_layers[res](x_)
            x_ += residual
            residual = x_

        out = x_
        return out


class DarkNet(nn.Module):
    def __init__(self, arch = [1,1,1,1,1]):
        super(DarkNet, self).__init__()
        print("\nImage Reduction factor by Convolutions 32\n")

        if isinstance(arch,list) or isinstance(arch, tuple):
            assert len(arch) == 5, "The darknet architecture requires 5 Blocks to be defined"
            res_units = arch
        else:
            raise Exception("Not sure what you are passing :(, DarkNet Class takes only iterable block unit count")

        self.block0 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1)   )

        self.block1 = DNBlock([32, 64], res_units[0])
        self.block2 = DNBlock([64, 128], res_units[1])
        self.block3 = DNBlock([128, 256], res_units[2])
        self.block4 = DNBlock([256, 512], res_units[3])
        self.block5 = DNBlock([512, 1024], res_units[4])

    def forward(self, x):
        x    = self.block0(x)
        x    = self.block1(x)
        x    = self.block2(x)
        out3 = self.block3(x)    # Largest FeatureMap
        out4 = self.block4(out3) # Medium Feature Map
        out5 = self.block5(out4) # Small Feature map

        return out3, out4, out5



class DetectorBlock(nn.Module):
    def __init__(self, in_chn, final_chn):
        super(DetectorBlock, self).__init__()

        self.detectnet = nn.Sequential(
        nn.Conv2d(in_chn, 512, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
        nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
        nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
        )

        self.outlayer = nn.Sequential(
        nn.Conv2d(1024, final_chn, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):

        u_out = self.detectnet(x)
        out = self.outlayer(u_out)
        return out, u_out


class YoloV3(nn.Module):
    def __init__(self, basenet_type = "darknet53", num_class = 80, num_anchor = 5):
        super(YoloV3, self).__init__()

        if basenet_type == "darknet53":
            basenet = DarkNet(arch=[1, 2, 8, 8, 4])
        elif basenet_type == "darknet21":
            basenet = DarkNet(arch=[1,1,2,2,1])
        else:
            raise Exception("Unknown Varient. Please pass the residual layers list manually through architecture.")


        self.final_channels = (5+num_class)*num_anchor

        self.backbone = basenet
        self.sf_detector = DetectorBlock( 1024,     self.final_channels)
        self.mf_detector = DetectorBlock( 512+1024, self.final_channels)
        self.lf_detector = DetectorBlock( 256+1024, self.final_channels)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):

        lf, mf, sf = self.backbone(x)

        sf_out, sf_feat = self.sf_detector(sf)
        sf_up = self.upscale(sf_feat)

        mf_in = torch.cat((mf, sf_up), 1)
        mf_out, mf_feat = self.mf_detector(mf_in)
        mf_up = self.upscale(mf_feat)

        lf_in = torch.cat((lf, mf_up), 1)
        lf_out, _ = self.lf_detector(lf_in)

        return sf_out, mf_out, lf_out


class YoloV3Inference():

    def __init__(self, model_fwd, anchors, classes,
                    conf_thresh = 0.6, nms_thresh = 0.5, device = "cpu"):
        self.model_fwd =  model_fwd
        self.anchors = anchors
        self.num_anchor = len(anchors)
        self.classes = classes
        self.num_class = len(classes)

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.device = device

    def run(self, x):

        res_tup = self.model_fwd(x)
        batch_sz = res_tup[0].shape[0]
        batch_result = []
        for bz in range(batch_sz):
            result = []
            for res in res_tup:
                res_ = res.view(res.shape[0], self.num_anchor,
                                5+len(self.classes),
                                res.shape[-2], res.shape[-1])
                result += self._get_box_info(res_[bz])

            result = self.apply_nms(result)
            batch_result.append(result)

        return res_tup, batch_result



    def _create_grid_corner_tnsr(self, shape):

        x_tnsr = torch.zeros(shape, requires_grad=False)
        y_tnsr = torch.zeros(shape, requires_grad=False)
        for i in range(shape[-1]):
            x_tnsr[...,:,i] = 1/shape[-1] * i
        for j in range(shape[-2]):
            y_tnsr[...,j,:] = 1/shape[-2] * i
        return x_tnsr.to(self.device), y_tnsr.to(self.device)

    def _create_anchor_hw_tnsr(self, shape):

        h_tnsr = torch.zeros(shape, requires_grad=False)
        w_tnsr = torch.zeros(shape, requires_grad=False)
        for i, anc in enumerate(self.anchors):
            h_tnsr[i,...] = anc[0]
            w_tnsr[i,...] = anc[1]
        return h_tnsr.to(self.device), w_tnsr.to(self.device)

    def _obtain_categ_score(self, ctnsr, top = 1):

        out = torch.topk(ctnsr, top)
        out_class = dict()
        for i, o in enumerate(out.indices):
            out_class[self.classes[o]] = float(out.values[i])
        return out_class

    def _get_box_info(self, tnsr):
        """
        Convert tensor to Bbox results per batch per anchor at a time
        tnsr: shape: [5+class, height, width]
        """
        conf_prd = torch.sigmoid(tnsr[:,0,:,:])

        y_gr, x_gr = tnsr.shape[-2], tnsr.shape[ -1]
        base_x_tnsr, base_y_tnsr = self._create_grid_corner_tnsr(conf_prd.shape)
        cx_prd = base_x_tnsr + torch.sigmoid(tnsr[:,1,:,:]) * (1/x_gr)
        cy_prd = base_y_tnsr + torch.sigmoid(tnsr[:,2,:,:]) * (1/y_gr)

        base_h_tnsr, base_w_tnsr = self._create_anchor_hw_tnsr(conf_prd.shape)
        h_prd = base_h_tnsr* (2 * torch.sigmoid(tnsr[:,3,:,:])) **3
        w_prd = base_w_tnsr* (2 * torch.sigmoid(tnsr[:,4,:,:])) **3

        categ_prd = torch.sigmoid(tnsr[:,5:,:,:])

        info_list = []
        for i in range(x_gr):
            for j in range(y_gr):
                for k in range(self.num_anchor):
                    if conf_prd[k,j,i] > self.conf_thresh:
                        odict = {
                            "conf": float(conf_prd[k,j,i]) ,
                            "cx": float(cx_prd[k,j,i]),
                            "cy": float(cy_prd[k,j,i]),
                            "h":  float(h_prd[k,j,i]),
                            "w":  float(w_prd[k,j,i]),
                            "x1": float(cx_prd[k,j,i] - (w_prd[k,j,i]/2)),
                            "y1": float(cy_prd[k,j,i] - (h_prd[k,j,i]/2)),
                            "x2": float(cx_prd[k,j,i] + (w_prd[k,j,i]/2)),
                            "y2": float(cy_prd[k,j,i] + (h_prd[k,j,i]/2)),
                            "categ_score": self._obtain_categ_score(categ_prd[k, :, j, i]),
                        }

                        info_list.append(odict)

        return info_list

    def apply_nms(self, data_list):

        pos_ = [ [d["x1"], d["y1"], d["x2"], d["y2"]] for d in data_list ]
        pos_tnsr = torch.tensor(pos_)
        if pos_tnsr.nelement() == 0: return []

        conf_ = [ d["conf"] for d in data_list]
        conf_tnsr = torch.tensor(conf_)

        res_idx = nms(pos_tnsr, conf_tnsr, self.nms_thresh)

        clean_list = [data_list[i] for i in res_idx ]

        return clean_list
