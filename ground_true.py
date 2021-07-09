import os
import cv2
# def get_ground_true(txt_path, classes_list):
'''
txt描述文件 image_name.jpg xmin ymin xmax ymax class
bottle 0.287150 336 231 376 305
'''

val_file_name = 'val.txt'
classes_list = ["今麦郎", "冰露", "百岁山", "怡宝", "百事可乐", "景甜", "娃哈哈", "康师傅", "苏打水", "天府可乐",
                    "可口可乐", "农夫山泉", "恒大冰泉", "其他"]
        

img_path = 'week10_dataset/image/'
output_file_dir = './input/ground-truth/'
image_write = './input/images-optional/'
result = []
file_path = os.path.join(img_path, val_file_name)
with open(file_path,'r',encoding='UTF-8') as f:
    lines = f.readlines()  # xxx.jpg xx xx xx xx class
    box = []
    label = []
    for line in lines:
        splited = line.strip().split()
        img_name = splited[0]  # 存储图片的地址+图片名称
        num_boxes = (len(splited) - 1) // 5  # 每一幅图片里面有多少个bbox


        image = cv2.imread(img_name)
        print(img_name)
#         print(image)

        tmp_len = len(img_name.split('/'))
        file_name = str(img_name.split('/')[tmp_len - 1])
        file_name = img_name.split('/')[tmp_len - 2]+file_name      

        cv2.imwrite(image_write+file_name,image)

        file_name = file_name.split('.')[0]
        output_predict_txt = output_file_dir + file_name + '.txt'

        print("############## write file %s##############" % (output_predict_txt))

        output_predict_file = open(output_predict_txt, 'w')

        for i in range(num_boxes):
            output_predict_file.write(
                classes_list[int(splited[5+5*i])] +
                ' ' +
                str(splited[1 + 5 * i]) +
                ' ' +
                str(splited[2 + 5 * i]) +
                ' ' +
                str(splited[3 + 5 * i]) +
                ' ' +
                str(splited[4 + 5 * i])
            )
            output_predict_file.write('\n')
        
    
        
