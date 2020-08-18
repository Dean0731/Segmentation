from util.dataset import dataset_tools
import tensorflow as tf
from PIL import Image
import os
import numpy as np
parent = "E:\Workspace\PythonWorkSpace\Segmentation\dataset"
# def getGernerator(batch_size,target_size=(576,576),mask_size=(576,576),num_classes=2):
#     with open(os.path.join(parent,'dom/data.txt'),encoding='utf-8') as f:
#         lines = f.readlines()
#     lines_x = []
#     lines_y = []
#
#     for k in lines:
#         lines_x.append(os.path.join(parent,k.strip().split(';')[0]))
#         lines_y.append(os.path.join(parent,k.strip().split(';')[1]))
#     i=0
#     while 1:
#         x_1 =[]
#         x_2 = []
#         y = []
#         for _ in range(batch_size):
#             image = Image.open(lines_x[i])
#             image_1 = image.resize(target_size)
#             image_1 = np.array(image_1)
#             x_1.append(image_1)
#
#             image_2 = image.resize((3072,3072))
#             image_2 = np.array(image_2)
#             x_2.append(image_2)
#
#             label = Image.open(lines_y[i])
#
#             label = label.resize(mask_size)
#             label = np.array(label)
#             new_label = np.zeros(label.shape + (num_classes,)).astype(int)  # (w,h,num_classes)
#             for m, n in zip(range(num_classes), [0,1]):
#                 new_label[:, :, m] = (n == label[:, :])
#             y.append(new_label)
#
#             if i == (len(lines)-1):
#                 i=0
#             else:
#                 i=i+1
#         yield ((np.array(x_1),np.array(x_2)),y)
#
#
# dataset = tf.data.Dataset.from_generator(getGernerator,output_types=((tf.int8,tf.int8),tf.int8),args=(4,2))
target_size = (512,512)
mask_size = (512,512)
num_classes = 2
batch_size = 2

dataset = dataset_tools.selectDataset('C3',"{}_{}".format('tif',3072),parent=parent)
dataset = dataset.getData(target_size=target_size,mask_size=mask_size,batch_size=batch_size)
for x,y in dataset:
    x_1,x_2=x
    print(len(x_1),len(x_2),len(y))
    print(x_1.shape)
    print(x_2.shape)
    print(y.shape)
    exit()



