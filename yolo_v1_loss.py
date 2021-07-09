# encoding:utf-8
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

yolo_v1_output = 24  # 5*2 + 14


def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred,target):
    return (pred-target)**2

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

class yoloLoss(nn.Module):  # (7, 2, 5, 0.5)
    def __init__(self, anchors, num_classes, img_size, lambda_conf = 4, lambda_cls = 3, lambda_loc = 1):
        # 为了更重视8维的坐标预测，给这些算是前面赋予更大的loss weight
        # 对于有物体的记为λ,coord，在pascal VOC训练中取5，
        # 对于没有object的bbox的confidence loss，前面赋予更小的loss weight 记为 λ,noobj, 在pascal VOC训练中取0.5
        # 有object的bbox的confidence loss (上图红色框) 和类别的loss （上图紫色框）的loss weight正常取1
        super(yoloLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.8
        self.lambda_conf = lambda_conf
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.anchors = [[28,160], [137,383]]
        self.img_size = [448, 448]

    def compute_iou(self, box1, box2):
        # iou 是求两个框的交并比
        # iou可用于求loss,可测试时的NMS

        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def box_ciou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # 求出预测框左上角右下角
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh/2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # 求出真实框左上角右下角
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 求真实框和预测框所有的iou
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area,min = 1e-6)

        # 计算中心的差距
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        # 找到包裹两个框的最小框的左上角和右下角
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # 计算对角线距离
        enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
        ciou = ciou - alpha * v
        return ciou
    def jaccard(self, _box_a, _box_b):
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
            box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
            box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)

        inter = inter[:, :, 0] * inter[:, :, 1]
        # 计算先验框和真实框各自的面积
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * \
             (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        # 求IOU
        union = area_a + area_b - inter
        return inter / union  # [A,B]
    def get_target(self, target, scaled_anchors, in_w, in_h, ignore_threshold):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(target)

        #-------------------------------------------------------#
        #   创建全是0或者全是1的阵列
        #-------------------------------------------------------#
        mask = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        noo_mask = torch.ones(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, int(self.num_anchors), in_h, in_w, 4, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False)
        for b in range(bs):
            if len(target[b])==0:
                continue
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            x2y2 = target[b][:, 2:4]
            x1y1 = target[b][:, :2]
            wh = x2y2 - x1y1
            cxcy = (x2y2 + x1y1) / 2

            target[b][:, 0] = cxcy[:,0]
            target[b][:, 1] = cxcy[:,1]
            target[b][:, 2] = wh[:,0]
            target[b][:, 3] = wh[:,1]

            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h
            #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
            anchors_index = [0, 1]
            #-------------------------------------------------------#
            #   计算出正样本相对于特征层的宽高
            #-------------------------------------------------------#
            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4]* in_h

            #-------------------------------------------------------#
            #   计算出正样本属于特征层的哪个特征点
            #-------------------------------------------------------#
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)
            
            #-------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            #-------------------------------------------------------#
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))
            
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   2, 4
            #-------------------------------------------------------#
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(scaled_anchors)), 1))
            #-------------------------------------------------------#
            #   计算交并比
            #   num_true_box, 2
            #-------------------------------------------------------#
            anch_ious = self.jaccard(gt_box, anchor_shapes)

            #-------------------------------------------------------#
            #   计算重合度最大的先验框是哪个
            #   num_true_box, 2
            #-------------------------------------------------------#
            best_ns = torch.argmax(anch_ious,dim=-1)
            for i, best_n in enumerate(best_ns):
                if best_n not in anchors_index:
                    continue
                #-------------------------------------------------------------#
                #   取出各类坐标：
                #   gi和gj代表的是真实框对应的特征点的x轴y轴坐标
                #   gx和gy代表真实框的x轴和y轴坐标
                #   gw和gh代表真实框的宽和高
                #-------------------------------------------------------------#
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]
#                 print('[b,i,gj,gi]',[b,i,gj,gi])
                tcls[b, gj, gi,target[b][i, 4].long()] = 1
                if (gj < in_h) and (gi < in_w):
                   
                    #----------------------------------------#
                    #   noo_mask代表无目标的特征点
                    #----------------------------------------#
                    noo_mask[b, best_n, gj, gi] = 0
                    #----------------------------------------#
                    #   mask代表有目标的特征点
                    #----------------------------------------#
                    mask[b, best_n, gj, gi] = 1
                    #----------------------------------------#
                    #   tx、ty代表中心的真实值
                    #----------------------------------------#
                    tx[b, best_n, gj, gi] = gx
                    ty[b, best_n, gj, gi] = gy
                    #----------------------------------------#
                    #   tw、th代表宽高的真实值
                    #----------------------------------------#
                    tw[b, best_n, gj, gi] = gw
                    th[b, best_n, gj, gi] = gh
                    #----------------------------------------#
                    #   用于获得xywh的比例
                    #   大目标loss权重小，小目标loss权重大
                    #----------------------------------------#
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]
                    #----------------------------------------#
                    #   tconf代表物体置信度
                    #----------------------------------------#
                    tconf[b, best_n, gj, gi] = 1
                    #----------------------------------------#
                    #   tcls代表种类置信度
                    #----------------------------------------#
                    # tcls[b, best_n, gj, gi, target[b][i, 4].long()] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue
        t_box[...,0] = tx
        t_box[...,1] = ty
        t_box[...,2] = tw
        t_box[...,3] = th
        return mask, noo_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y, target

    def get_ignore(self, predict_local, target_tensor, scaled_anchors, in_w, in_h, noo_mask):
        # noo_mask: torch.Size([2, 14, 14])
        #-----------------------#
        #   一共多少张图片-------#
        #-----------------------#
        bs = predict_local.size(0)
#         predict_local = predict_local.view(bs, in_h, in_w, int(self.num_anchors),5).permute(0, 3, 1, 2, 4).contiguous()
        
        predict_local1 = predict_local[...,:5]
        predict_local1 = predict_local1.unsqueeze(1)
        predict_local2 = predict_local[...,5:10]
        predict_local2 = predict_local2.unsqueeze(1)
        
#         print('predict_local1.size',predict_local1.size())
#         print('predict_local2.size',predict_local2.size())
        predict_local = torch.cat([predict_local1, predict_local2], axis=1)
#         print('predict_local.size',predict_local.size())
        

        # 先验框的中心位置的调整参数
        # predict_local.size():  torch.Size([2, 2, 14, 14, 5])      (10+14)
        x = torch.sigmoid(predict_local[..., 0])
        y = torch.sigmoid(predict_local[..., 1])
        # 先验框的宽高调整参数
        w = predict_local[..., 2]  # Width
        h = predict_local[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs*self.num_anchors), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs*self.num_anchors), 1, 1).view(y.shape).type(FloatTensor)

        
#         print('grid_x',grid_x)
#         print('grid_y',grid_y)
        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        
#         print('')
#         print('anchor_h.size:',anchor_h.size())
#         print('anchor_h:',anchor_h)
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
#         print('anchor_h.size():',anchor_h.size())
#         print('anchor_h:',anchor_h)
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #-------------------------------------------------------#
        pred_boxes = FloatTensor(predict_local[..., :4].shape)
        # [2,2,14,14,4]
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i] # [2,14,14,4]
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(target_tensor[i]) > 0:
                gx = target_tensor[i][:, 0:1] * in_w
                gy = target_tensor[i][:, 1:2] * in_h
                
                
                gw = target_tensor[i][:, 2:3] * in_w                
                gh = target_tensor[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh],-1)).type(FloatTensor)

                #-------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = self.jaccard(gt_box, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
#                 print('anch_ious.size:',anch_ious.size())
                anch_ious_max, _ = torch.max(anch_ious,dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3]) # [2,14,14,4]
                noo_mask[i][anch_ious_max>self.ignore_threshold] = 0
            return noo_mask, pred_boxes
# noo_mask: torch.Size([2,2 , 14, 14])

    def forward(self, pred_tensor, target_tensor):  # s-size=14 B-boxcount=2
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        
        # 坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例

        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        '''
        # print('pred_tensor.size(): ', pred_tensor.size())
        # print('target_tensor.size(): ', target_tensor.size())
#         import pdb
#         pdb.set_trace()
        bs = pred_tensor.size(0)
        #-----------------------#
        #   特征层的高
        #-----------------------#
        in_h = pred_tensor.size(1)
        #-----------------------#
        #   特征层的宽
        #-----------------------#
        in_w = pred_tensor.size(2)
        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   如果特征层为14x14的话，一个特征点就对应原来的图片上的448/14=32个像素点
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------------------------------#
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        
        
        # N = pred_tensor.size()[0]  # batch-size N=3
        # target_tensor = encode(target_tensor)
        # coo_mask = target_tensor[:, :, :, 4] > 0  # 具有目标标签的索引
        # noo_mask1 = target_tensor[:, :, :, 4] == 0  # 不具有目标的标签索引
#         noo_mask = torch.ByteTensor(noo_mask1.size())
#         noo_mask.zero_()
        
        # tcls[b, best_n, gj, gi, target[b][num_anchors, 4].long()] = 1
        #  t_box 计算出正样本相对于特征层的宽高,及中心点
        
        # target_tensor = #
        # 获得置信度，是否有物体
#         pred_tensor
        conf = torch.sigmoid(pred_tensor[..., [4,9]])
        conf = conf.permute(0, 3, 1, 2).contiguous()  # 2,14,14,2
        
        # 种类置信度
        pred_cls = torch.sigmoid(pred_tensor[..., 10:])

        coo_mask, noo_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y, target =self.get_target(target_tensor, scaled_anchors,in_w, in_h,self.ignore_threshold)
        #[2,2,14,14]

        # pred_boxes   [2,2,14,14,4]
        pred_local = pred_tensor[...,:10]
        noo_mask, pred_boxes_for_ciou = self.get_ignore(pred_local, target, scaled_anchors, in_w, in_h, noo_mask)

        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        #---------------------------------------------------------------#
        #   计算预测结果和真实结果的CIOU
        #----------------------------------------------------------------#
#         ciou = (1 - self.box_ciou( pred_boxes_for_ciou[coo_mask.bool()], t_box[coo_mask.bool()]))* box_loss_scale[coo_mask.bool()]
        ciou = (1 - self.box_ciou( pred_boxes_for_ciou[coo_mask.bool()], t_box[coo_mask.bool()]))
        loss_loc = torch.sum(ciou)

        # 计算置信度的loss
        loss_conf = torch.sum(BCELoss(conf, coo_mask) * coo_mask) + \
                    torch.sum(BCELoss(conf, coo_mask) * noo_mask)
                    
        mask = torch.zeros(bs, in_h, in_w, requires_grad=False)
        for b in range(bs):  
            for i in range(in_h):
                for j in range(in_w):
                    mask[b,i,j] = coo_mask[b,0,i,j] or coo_mask[b,1,i,j]
                    
#         print('mask',mask)
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]))
        
        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc +1e-8
#         print("loss_conf= ", loss_conf.item())  # 5
#         print("loss_loc= ", loss_loc.item())  #
#         print("loss_cls= ", loss_cls.item())  #
#         print("loss= ", loss.item())
        return loss, loss_conf, loss_loc, loss_cls
        









