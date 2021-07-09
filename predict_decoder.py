# encoding:utf-8
#
# created by xiongzihua
#
import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
from resnet_yolo_v1 import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np
import pdb
import os

classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
                "可口可乐", "农夫山泉", "恒大冰泉", "其他"]
anchors = [[28,160], [137,383]]

def jaccard( _box_a, _box_b):
        _box_a = _box_a.view(-1,4)
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = _box_a[:, 0], _box_a[:, 1], _box_a[:, 2], _box_a[:, 3]
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = _box_b[:, 0], _box_b[:, 1], _box_b[:, 2], _box_b[:, 3]
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
def nms(bboxes, scores, confidences, threshold=0.4):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    # pdb.set_trace()
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
#     _, index = scores.sort(0, descending=True)
    _, index = confidences.sort(0, descending=True)
    print(index)
    order_array = index.numpy()
    print(order_array)
    index = order_array
    print(index)
    index_len = len(index)
    print(index_len)
    keep = []
    
    index = index[:30]
    # while index.numel() > 0:
    
#     while index_len > 0:
    while len(index)>0:
        index_len = index_len - 1
        i = index[0]  # every time the first is the biggst, and add it directly
        print('get i:',i)
        keep.append(i)
        ious = jaccard(bboxes[i], bboxes[index[1:]])

        # pdb.set_trace()
        print("index[%d] ious=" % (i))
        print(ious)
        idx = np.where(ious <= threshold)[1]
        print(idx)
        index = index[idx + 1]  # because index start from 1

    print(keep)
    return torch.LongTensor(keep)
  
  

def decoder(pred, imageSize):
    '''
    pred (tensor)
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    # pdb.set_trace()

    grid_num = 14
    boxes = []
    cls_indexs = []
    probs = []
    confidences = []
    
    pred[...,4] = torch.sigmoid(pred[...,4]) 
    pred[...,9] = torch.sigmoid(pred[...,9])
    pred[...,0:2] = torch.sigmoid(pred[...,0:2])
    pred[...,5:7] = torch.sigmoid(pred[...,5:7])
    pred[...,10:] = torch.sigmoid(pred[...,10:])

    #-----------------------------------------------#
    #   输入的input的shape
    #-----------------------------------------------#

    input_height = pred.size(1)
    input_width = pred.size(2)
    print('pred.size:',pred.size())
    #-----------------------------------------------#
    #   输入为448*448时
    #   stride_h = stride_w = 32、16、8
    #-----------------------------------------------#
    stride_h = imageSize[0] / input_height
    stride_w = imageSize[1] / input_width

    #-------------------------------------------------#
    #   此时获得的scaled_anchors大小是相对于特征层的
    #-------------------------------------------------#
    scaled_anchors = [[anchor_width / stride_w, anchor_height / stride_h] for anchor_width, anchor_height in anchors]

    FloatTensor = torch.cuda.FloatTensor if pred[...,0].is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if pred[...,0].is_cuda else torch.LongTensor

    #----------------------------------------------------------#
    #   按照网格格式生成先验框的宽高
    #   
    #----------------------------------------------------------#
#     anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
#     anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
#     anchor_w = anchor_w.repeat(1, 1)
#     anchor_h = anchor_h.repeat(1,1)


    anchor_w = scaled_anchors[:][0]
    anchor_h = scaled_anchors[:][1]
    print('anchor_w:',anchor_w)
    print('anchor_h:',anchor_h)

    #----------------------------------------------------------#
    #   利用预测结果对先验框进行调整
    #   首先调整先验框的中心，从先验框中心向右下角偏移
    #   再调整先验框的宽高。
    #----------------------------------------------------------#

    #----------------------------------------------------------#
    #   将输出结果调整成相对于输入图像大小的比例
    #----------------------------------------------------------#
    _scale = torch.Tensor([1/ input_width,1/input_height] * 2).type(FloatTensor)

    print('pred.size: ', pred.size())
    pred = pred.data
    pred = pred.squeeze(0)  #
    
    print('pred.size: ', pred.size())
    contain = pred[:, :, [4,9]]

    mask1 = contain > 0.3  # 大于阈值
    mask2 = (contain == contain.max())  # we always select the best contain_prob what ever it>0.9pred[:,:,4]
    mask = (mask1 + mask2).gt(0)  #获得大于0 的
    
#     print('contain.size: ', contain.size())
#     print('contain :', contain.data)
#     print('mask1.size: ', mask1.size())
#     print('mask1 :', mask1.data)
#     print('mask2.size: ', mask2.size())
#     print('mask2 :', mask2.data)
#     print('mask.size: ', mask.size())
#     print('mask :', mask.data)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
           
#             b = 0 if pred[i, j, 4] > = pred[i, j, 9]  else 1
            for b in range(2):
                # print("b=%d" %(b))
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    print("mask[%d,%d,%d]==1" % (i, j, b))
                    # print(i,j,b)
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    box[0] = box[0] + j
                    box[1] = box[1] + i
                    box[2] = torch.exp(box[2]) * anchor_w[b]
                    box[3] = torch.exp(box[3]) * anchor_h[b]
                    box = box * _scale
                    print('box.size:',box.size())


                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)

                    print("get a box")
                    print(box_xy)
                    print("the box cls=")
                    print(cls_index)
                    print("the box max_prob")
                    print(max_prob)
                    print("the box contain_prob")
                    print(contain_prob)
                    print("float((contain_prob*max_prob)[0])")
                    print(float((contain_prob * max_prob)[0]))
                    ENABLE_VALUE = 0.02
                    # ENABLE_VALUE = 0.02

                    # if float((contain_prob*max_prob)[0]) > ENABLE_VALUE:
                    if float((contain_prob * max_prob)[0]) > ENABLE_VALUE:

                        print("find a box (%d %d %d)" % (i, j, b))
                        boxes.append(box_xy.view(1, 4))
                        # pdb.set_trace()

                        tmp_list = []
                        tmp_int = cls_index.item()
                        tmp_list.append(tmp_int)
                        tmp_tensor = torch.tensor(tmp_list)
                        cls_indexs.append(tmp_tensor)
                        # cls_indexs.append(cls_index)

                        # cls_indexs.append(float(cls_index.item()))

                        confidences.append(contain_prob)
                        probs.append(contain_prob * max_prob)
                    else:
                        print("contain_prob*max_prob not > 0.02")
                # else:
                # print("mask not 1")
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.ones(1)
        cls_indexs = torch.zeros(1)
        confidences = torch.ones(1)

    else:
        # pdb.set_trace()
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        confidences = torch.cat(confidences, 0)
        # pdb.set_trace()
        # print(cls_indexs)
        cls_indexs_len = len(cls_indexs)
        print("cls_indexs_len=%d" % (cls_indexs_len))

        # cls_indexs = torch.cat(torch.tensor(cls_indexs),0) #(n,)
        cls_indexs = torch.cat(cls_indexs, 0)  # (n,)

    keep = nms(boxes, probs, confidences)
    return boxes[keep], cls_indexs[keep], probs[keep], confidences[keep]

def predict(model, image_name, root_path=''):
    result = []
    imageSize = (448, 448)
   
    print("img name=%s" % (image_name))
    image = cv2.imread(image_name)
    
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (141, 132, 126)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = Variable(img[None, :, :, :], volatile=True)
    img = img

    pred = model(img)
    
    pred = pred.cpu()


    # return (tensor) box[[x1,y1,x2,y2]] label[...]
    boxes, cls_indexs, probs, confidences = decoder(pred, imageSize)

    for i, box in enumerate(boxes):
        if probs[i] > 0.02:
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index)  # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            confidence = confidences[i]
            confidence = float(confidence)
            result.append([(x1, y1), (x2, y2), classes_list[cls_index], image_name, confidence])
    return result


def write_output_file(result, output_file_dir):
    i = 0
    for left_up, right_bottom, class_name, _, prob in result:
        # pdb.set_trace()
#         print("result index=%d" % (i))
#         print(result)
        if i == 0:
            tmp_len = len(_.split('/'))
            file_name = _.split('/')[tmp_len - 1]
            
            file_name = img_name.split('/')[tmp_len - 2]+file_name
            
            file_name = file_name.split('.')[0]
            output_predict_txt = output_file_dir + file_name + '.txt'

            print("############## write file %s##############" % (output_predict_txt))

            output_predict_file = open(output_predict_txt, 'w')

        output_predict_file.write(
            str(class_name) +
            ' ' +
            str(prob) +
            ' ' +
            str(left_up[0]) +
            ' ' +
            str(left_up[1]) +
            ' ' +
            str(right_bottom[0]) +
            ' ' +
            str(right_bottom[1]) +
            '\n'
        ) 

        i = i + 1


if __name__ == '__main__':
#     pass
    val_file_name = 'val.txt'
    output_file_dir = './input/detection-results/'
    img_path = './week10_dataset/image/'
    file_path = os.path.join(img_path, val_file_name)
    
    # network structure
    model = resnet50()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)


    
#     model.load_state_dict(torch.load('./weight/2:test.pth'))
    model.load_state_dict(torch.load('./weight/5:test.pth'))
   
    
    model.eval
    with open(file_path) as f:
        lines = f.readlines()  # xxx.jpg xx xx xx xx class
        box = []
        label = []
        for line in lines:
            splited = line.strip().split()
            img_name = splited[0]  # 存储图片的地址+图片名称
            result = predict(model, img_name )
            write_output_file(result, output_file_dir)
            
 




