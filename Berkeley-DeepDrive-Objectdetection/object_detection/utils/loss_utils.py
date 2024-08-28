import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np

class YoloV3Loss():
    '''
    Yolo loss for 2d Bounding box
    '''
    def __init__(self, classes , anchors, scale_anchor =1, iou_thresh = 0.0, device = "cpu"):
        self.scale_anchor = scale_anchor
        self.anchors = [(h/self.scale_anchor, w/self.scale_anchor)
                        for h,w in anchors ]
        self.num_anchor = len(anchors)
        self.num_class = len(classes)
        self.iou_thresh = iou_thresh
        self.device = device

        self.λcoord = 5
        self.λnoobj = 0.5

    def compute_loss(self, pred_tnsr_tup, truth_dict):

        loss = 0
        for tnsr in pred_tnsr_tup:
            # tnsr_ :shape: [batch, anchors, class+5, h,w]
            tnsr_ = tnsr.view(tnsr.shape[0], self.num_anchor, 5+self.num_class,
                              tnsr.shape[-2], tnsr.shape[-1])

            truth_tnsr = self.contruct_truth_tnsr(truth_dict, tnsr_.shape)

            loss += self.confidence_loss(tnsr_, truth_tnsr)
            loss += self.bbox_pos_loss(tnsr_, truth_tnsr)
            loss += self.bbox_hw_loss(tnsr_, truth_tnsr)
            loss += self.category_loss(tnsr_, truth_tnsr)

        return loss

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

    def contruct_truth_tnsr(self, trgtd, tnsr_shape):
        '''
        Return:
            obj_mask_tnsr: Mask tensor corresponding to anchors, used for masking loss computation
            pos_base_tnsr: true_pos - anchor_pos
        prd_tnsr :shape: [batch, anchors, class+5, h,w]
        '''
        batch_sz, y_gr, x_gr = tnsr_shape[0], tnsr_shape[-2], tnsr_shape[-1]
        truth_tnsr = torch.zeros(tnsr_shape, requires_grad=False)

        ## Create Objectness tensor
        for b in range(batch_sz):

            # associate Grid
            for trg in trgtd[b]:

                cx = trg['cx']; cy = trg['cy']
                x1 = trg['x1']; y1 = trg['y1']
                x2 = trg['x2']; y2 = trg['y2']
                h = trg['h']; w = trg['w']
                cid = trg['categ_id']

                # Map relative-location to grid
                i,j = int(cx*x_gr), int(cy*y_gr)

                # compute grid corner pixel location relative
                gx, gy = i/x_gr, j/y_gr

                # Associate AnchorBox based on IoU
                iou = [0]*self.num_anchor
                for k in range(self.num_anchor):
                    ah, aw = self.anchors[k]
                    iou[k] = self.IoU(  {'x1':x1,'y1':y1, 'x2':x2,'y2':y2 },
                                        {'x1':gx -(aw/2),'y1':gy -(ah/2),
                                         'x2':gx +(aw/2),'y2':gy +(ah/2)}, )
                #
                kt = [iou.index(i) for i in iou if i>self.iou_thresh]
                # print("IOUed Anchors:", kt, iou)
                truth_tnsr[b,kt,0,j,i] = 1
                truth_tnsr[b,kt,1,j,i] = cx # true_cx
                truth_tnsr[b,kt,2,j,i] = cy # true_cy
                truth_tnsr[b,kt,3,j,i] = h  # true_h
                truth_tnsr[b,kt,4,j,i] = w  # true_w
                truth_tnsr[b,kt,int(cid),j,i] = 1
            #
        ##
        return truth_tnsr.to(self.device)

    def confidence_loss(self, prd_tnsr, true_tnsr):
        true_conf_tnsr = true_tnsr[:,:,0,:,:]
        objm = true_tnsr[:,:,0,:,:].bool()
        noobjm = ~objm
        prd_conf_tnsr = prd_tnsr[:,:,0,:,:]

        obj_loss = nnf.binary_cross_entropy_with_logits(prd_conf_tnsr[objm],
                                            true_conf_tnsr[objm],
                                            reduction = 'sum')
        noobj_loss = nnf.binary_cross_entropy_with_logits(prd_conf_tnsr[noobjm],
                                                true_conf_tnsr[noobjm],
                                                reduction = 'mean')
        obj_cnt = torch.sum(objm.int())
        loss = obj_loss + self.λnoobj * max(1,obj_cnt) * noobj_loss

        return loss

    def _create_grid_corner_tnsr(self, shape):

        x_tnsr = torch.zeros(shape, requires_grad=False)
        y_tnsr = torch.zeros(shape, requires_grad=False)
        for i in range(shape[-1]):
            x_tnsr[...,:,i] = 1/shape[-1] * i
        for j in range(shape[-2]):
            y_tnsr[...,j,:] = 1/shape[-2] * i
        return x_tnsr.to(self.device), y_tnsr.to(self.device)

    def bbox_pos_loss(self, prd_tnsr, true_tnsr):

        y_gr, x_gr = true_tnsr.shape[-2], true_tnsr.shape[ -1]
        objm = true_tnsr[:,:,0,:,:].bool()
        true_cx_tnsr = true_tnsr[:,:,1,:,:]
        true_cy_tnsr = true_tnsr[:,:,2,:,:]
        base_x_tnsr, base_y_tnsr = self._create_grid_corner_tnsr(true_cy_tnsr.shape)

        prd_tx_tnsr = torch.sigmoid(prd_tnsr[:,:,1,:,:]) * (1/x_gr)
        prd_ty_tnsr = torch.sigmoid(prd_tnsr[:,:,2,:,:]) * (1/y_gr)

        prd_cx_tnsr = base_x_tnsr + prd_tx_tnsr
        prd_cy_tnsr = base_y_tnsr + prd_ty_tnsr

        pos_loss_x =  nnf.mse_loss( true_cx_tnsr[objm], prd_cx_tnsr[objm],
                                    reduction = 'sum')
        pos_loss_y = nnf.mse_loss(  true_cy_tnsr[objm], prd_cy_tnsr[objm],
                                    reduction = 'sum')
        pos_loss = self.λcoord * (pos_loss_x + pos_loss_y)

        return pos_loss

    def _create_anchor_hw_tnsr(self, shape):

        h_tnsr = torch.zeros(shape, requires_grad=False)
        w_tnsr = torch.zeros(shape, requires_grad=False)
        for i, anc in enumerate(self.anchors):
            h_tnsr[:,i,...] = anc[0]
            w_tnsr[:,i,...] = anc[1]
        return h_tnsr.to(self.device), w_tnsr.to(self.device)

    def bbox_hw_loss(self, prd_tnsr, true_tnsr):

        objm = true_tnsr[:,:,0,:,:].bool()
        true_h_tnsr =  torch.sqrt(true_tnsr[:,:,3,:,:])
        true_w_tnsr =  torch.sqrt(true_tnsr[:,:,4,:,:])
        base_h_tnsr, base_w_tnsr = self._create_anchor_hw_tnsr(true_h_tnsr.shape)

        # exp as in Yolov3 paper causes instability while training,
        # Using PowerMethod-> (2*σ(x))^3; purposed by Ultralytics Yolov3
        prd_h_tnsr = base_h_tnsr* (2 * torch.sigmoid(prd_tnsr[:,:,3,:,:])) **3
        prd_w_tnsr = base_w_tnsr* (2 * torch.sigmoid(prd_tnsr[:,:,4,:,:])) **3
        prd_h_tnsr = torch.sqrt(prd_h_tnsr)
        prd_w_tnsr = torch.sqrt(prd_w_tnsr)

        sz_loss_h = nnf.mse_loss(true_h_tnsr[objm], prd_h_tnsr[objm],
                                reduction = 'sum')
        sz_loss_w = nnf.mse_loss(true_w_tnsr[objm], prd_w_tnsr[objm],
                                reduction = 'sum')

        sz_loss = self.λcoord * (sz_loss_h + sz_loss_w)

        return sz_loss

    def category_loss(self, prd_tnsr, true_tnsr):

        true_categ_tnsr = true_tnsr[:,:,5:,:,:]
        num_class = true_categ_tnsr.shape[2]
        #objm :shape: b,ac,1,h,w
        objm = true_tnsr[:,:,0:1,:,:].bool()
        #objm :shape: b,ac,num_class,h,w
        objm = objm.repeat(1,1,num_class,1,1)

        prd_categ_tnsr = prd_tnsr[:,:,5:,:,:]

        loss = nnf.binary_cross_entropy_with_logits( prd_categ_tnsr[objm],
                                         true_categ_tnsr[objm],
                                         reduction = 'sum' )

        return loss

