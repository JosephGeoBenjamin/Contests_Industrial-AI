import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np


class ObjectDetectAccuracy():
    def __init__(self, iou_thresh):
        self.precision = 0
        self.recall = 0
        self.f1score = 0

        self.iou_thresh = iou_thresh

    def compute(self, truth_batch, pred_batch):
        b_sz = len(truth_batch)
        precision_, recall_, f1score_ = 0,0,0

        for truth_dict, pred_dict in zip(truth_batch, pred_batch):
            if pred_dict:
                TP, FP, FN = self.calc_trueness(truth_dict, pred_dict)
                precision_ += len(TP) / len(pred_dict)  # TP+FP = pred
                recall_ += len(TP) / len(truth_dict)    # TP+FN = truth
                f1score_ += (2 * precision_ * recall_) / (precision_+recall_)

        return precision_ / b_sz, recall_ / b_sz, f1score_ /b_sz

    def update(self, truth_batch, pred_batch):

        p,r,f = self.compute( truth_batch, pred_batch)
        self.precision += p
        self.recall += r
        self.f1score += f

    def get(self, divideby = 1):
        return (    self.precision / divideby,
                    self.recall / divideby,
                    self.f1score / divideby
                )

    def reset(self):
        self.__init__()


    def IoU(self, r_a, r_b):
        """
        r_a, r_b: {x1:,y1:,x2:,y2:}
        """
        anb_w = max(0, min(r_a['x2'],r_b['x2']) - max(r_a['x1'],r_b['x1']) )
        anb_h = max(0, min(r_a['y2'],r_b['y2']) - max(r_a['y1'],r_b['y1']) )
        anb = anb_h * anb_w
        aub = (r_a['x2'] - r_a['x1'])*(r_a['y2'] - r_a['y1']) + \
                (r_b['x2'] - r_b['x1'])*(r_b['y2'] - r_b['y1']) - anb

        return anb / aub

    def calc_trueness(self, trdc, prdc):
        tp_ = []
        fn_ = []
        for t in trdc:
            tp_select = None
            for p in prdc:
                if t["category"] in p["categ_score"].keys():
                    if IoU(t , p) > self.iou_thresh:
                        tp_select = p
            if tp_select:
                tp_.append(tp_select)
                prdc.remove(tp_select)
            else:
                fn_.append(t)
        fp_ = prdc

        return tp_, fp_, fn_