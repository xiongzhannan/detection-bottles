#coding:utf-8
import numpy as np
import random
from tqdm import tqdm
from Week10Dataset_process import myDataset

from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from resnet_yolo_v1 import resnet50
from yolo_v1_loss import yoloLoss
from predict_decoder import decoder
import cv2
import torch




device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
                "可口可乐", "农夫山泉", "恒大冰泉", "其他"]
Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


cuda = True
retrain = False
img_path = 'week10_dataset/image/'
batch_size = 2
learning_rate = 0.0001
num_epochs = 200
prepoch = 0
no_get_better_epoch = 0

anchors= [[28,160], [137,383]]
num_classes, input_shape = len(classes_list), (448,448)

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# train dataset
train_dataset = myDataset(img_path=img_path, file_name='train.txt', train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
# test dataset
test_dataset = myDataset(img_path=img_path, file_name='val.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)

# network structure
net = resnet50()
net = net.to(device)
if not retrain:
    # load pre_trained model
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    
    op = net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in op.keys() and not k.startswith('fc'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
            print('yes')
            op[k] = new_state_dict[k]
    net.load_state_dict(op)
    #
    prepoch = 0
    rest_epoch = num_epochs - prepoch
else:
    # load trained model

    net.load_state_dict(torch.load('15_1:best.pth'))
    
    prepoch = 0
    rest_epoch = num_epochs - prepoch
    learning_rate = 0.0002



criterion = yoloLoss(anchors, num_classes, (input_shape[1], input_shape[0]), lambda_conf = 3, lambda_cls = 1, lambda_loc = 6)

# different learning rate
params = []
params_dict = dict(net.named_parameters())

for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]

# optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.99), eps=1e-08,)

torch.multiprocessing.freeze_support()
file_path = img_path +'train.txt'
with open(file_path) as f:
    lines = f.readlines()

best_train_loss = np.inf
best_test_loss = np.inf
num_train = len(lines)
epoch_size = max(1, num_train//batch_size)

val_path = img_path +'val.txt'
with open(val_path) as val_f:
    val_lines = val_f.readlines()
num_val = len(val_lines)
epoch_size_val = max(1, num_val//batch_size)

for epoch in range(rest_epoch):
    real_epoch =epoch + prepoch
    net.train()
    if real_epoch == 10:
        learning_rate = 0.005
    
    if real_epoch == 30:
        learning_rate = 0.0001
    if real_epoch == 50:
        learning_rate = 0.005
    if real_epoch == 70:
        learning_rate = 0.001
    if real_epoch == 100:
        learning_rate = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    

    with tqdm(total=epoch_size,desc=f'Epoch {real_epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.1) as pbar:
        net.train()
        one_epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            if i >= epoch_size:
                    break
            # images, target = images.cuda(), target.cuda() # use gpu
            images = batch[0]
            targets = batch[1]
    #         if cuda:
    #             images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
    #             targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
    #         else:
    #             images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
    #             targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            # print('images.shape:', images.shape)
            images = torch.from_numpy(images).type(torch.FloatTensor)
            targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            images = images.permute(0, 3, 1, 2).contiguous()
            # print('images.size:', images.size())
            # print('targets.shape:', len(targets))
            optimizer.zero_grad()

            pred = net(images)
            # print('pred.size(): ', pred.size())

            loss, loss_conf, loss_loc, loss_cls = criterion(pred, targets)
            one_epoch_loss +=loss.item()

            
            loss.backward()
            optimizer.step()
            # if (i + 1) % 5 == 0:
            #         print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), one_epoch_loss / (i + 1)))
            pbar.set_postfix({'loss_conf': loss_conf.item()/batch_size,'loss_loc': loss_loc.item()/batch_size,'loss_cls': loss_cls.item()/batch_size,'average_loss': one_epoch_loss/(i + 1)/batch_size,'lr': learning_rate})
            pbar.update(1)
    
    if real_epoch+1>10: no_get_better_epoch +=1
    if real_epoch+1 == 10: 
        best_train_loss = one_epoch_loss/epoch_size/batch_size
        print('get best train loss %.5f' % best_train_loss)
#         torch.save(net.state_dict(), './train_weight/%d:train.pth'% ((real_epoch+1)/10))
    
    
    if real_epoch+1 > 10 and best_train_loss> one_epoch_loss/epoch_size/batch_size:
        best_train_loss = one_epoch_loss/epoch_size/batch_size
        print('get best train loss %.5f' % best_train_loss)
        torch.save(net.state_dict(), './train_weight/%d:train.pth'% ((real_epoch+1)/10))
        
        no_get_better_epoch = 0
         
        

        
    if real_epoch+1 > 10:
        print('++++++Start Validation++++++')
        with tqdm(total=epoch_size_val, desc=f'Epoch {real_epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.1) as pbar:
            net.eval()
            one_epoch_test_loss = 0.0
            for iteration, batch in enumerate(test_loader):
                if iteration >= epoch_size_val:
                    break
                images_val, targets_val = batch[0], batch[1]
                
                images = torch.from_numpy(images_val).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                images = images.permute(0, 3, 1, 2).contiguous()
                # print('images.size:', images.size())
                # print('targets.shape:', len(targets))
                optimizer.zero_grad()

                pred = net(images)
                # print('pred.size(): ', pred.size())

                loss, loss_conf, loss_loc, loss_cls = criterion(pred, targets)
                one_epoch_test_loss +=loss.item()


                pbar.set_postfix(**{'loss_conf': loss_conf.item()/batch_size,
                                'loss_loc': loss_loc.item()/batch_size,
                                'loss_cls': loss_cls.item()/batch_size,
                                'test_loss': one_epoch_test_loss/(iteration + 1)/batch_size})
                pbar.update(1)
        print('Finish Validation')
        print('Epoch:'+ str(real_epoch+1) + '/' + str(num_epochs))
        print('++++++Train Loss: %.4f || Val Loss: %.4f ++++++' % (one_epoch_loss/(epoch_size)/batch_size,one_epoch_test_loss/(epoch_size_val)/batch_size))
        if real_epoch+1 == 11: 
            best_test_loss = one_epoch_test_loss/epoch_size_val/batch_size
            print('get best test loss %.5f' % best_test_loss)

 
        if real_epoch+1 > 11 and best_test_loss > one_epoch_test_loss/epoch_size_val/batch_size:
            print('Saving state, iter:', str(real_epoch+1))
            best_test_loss = one_epoch_test_loss/epoch_size_val/batch_size
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), './test_weight/%d:test.pth'% ((real_epoch+1)/10))
            
            no_get_better_epoch = 0
        print('=================================================================')
        print('')
    if no_get_better_epoch >=30: break




# predict
val_file_name = 'val.txt'
result = []
file_path = os.path.join(img_path, val_file_name)
with open(file_path) as f:
    lines = f.readlines()  # xxx.jpg xx xx xx xx class
    box = []
    label = []
    for line in lines:
        splited = line.strip().split()
        img_name = splited[0]  # 存储图片的地址+图片名称
        image = cv2.imread(img_name)
        h, w, _ = image.shape
        img = cv2.resize(image, (448, 448))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = (141, 132, 126)  # RGB

        img = img - np.array(mean, dtype=np.float32)

        transform = transforms.Compose([transforms.ToTensor(), ])
        img = transform(img)
        img = Variable(img[None, :, :, :], volatile=True)
        #             img = img.cuda()

        pred = net(img)  # 1x14x14x24
        pred = pred.cpu()

        boxes, cls_indexs, probs = decoder(pred, imageSize = (448, 448))

        for i, box in enumerate(boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index)  # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            result.append([(x1, y1), (x2, y2), classes_list[cls_index], img_name, prob])



for left_up, right_bottom, class_name, img_name, prob in result:
    image = cv2.imread(img_name)
    output_name = class_name
    color = Color[classes_list.index(class_name)]
    cv2.rectangle(image, left_up, right_bottom, color, 2)
    label = class_name + str(round(prob, 2))
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    p1 = (left_up[0], left_up[1] - text_size[1])
    cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                  color, -1)
    cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    
#     cv2.imshow(output_name, image)


#     validation_loss = 0.0
    
#     best_test_loss = np.inf
#     net.eval()

#     for i, (images, target) in enumerate(test_loader):
#         # images, target = images.cuda(), target.cuda()
#         pred = net(images)
#         loss = criterion(pred, target)
#         validation_loss += loss.item()
#     validation_loss /= len(test_loader)


#     if best_test_loss > validation_loss:
#         best_test_loss = validation_loss
#         print('get best test loss %.5f' % best_test_loss)
#         torch.save(net.state_dict(), 'best.pth')
#     torch.save(net.state_dict(), 'yolo.pth')

