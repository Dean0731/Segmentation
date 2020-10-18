# @Time     : 2020/10/7 11:42
# @File     : Transform
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/10/7 Dean First Release

from tf.util.Config import tf
from tf.util.Config import config
from PIL import Image
import numpy as np
# 数据集转换 普通转换
# SparseCategoricalCrossentropy
def transform_common(line_x,line_y):
    image = tf.io.read_file(line_x)
    image = tf.io.decode_png(image)
    image = tf.image.resize(image,config.target_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = image
    label = tf.io.read_file(line_y)
    label = tf.io.decode_png(label,channels=1)
    label = tf.image.resize(label,config.mask_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = np.array(label).astype(np.float32)
    label[label==38]=1
    label = tf.convert_to_tensor(label)
    y = label
    return x,y
# 双输入
def transform_double_input(line_x,line_y):
    image = tf.io.read_file(line_x)
    image = tf.image.decode_png(image)
    image_1 = tf.image.resize(image,config.target_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_2 = tf.image.resize(image,(3072,3072),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x_1,x_2 = image_1,image_2

    label = tf.io.read_file(line_y)
    label = tf.image.decode_png(label,channels=1)
    label = tf.image.resize(label,config.mask_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.squeeze(label)
    label = tf.cast(label,dtype=tf.uint8)
    label = tf.one_hot(label,depth=2)
    label = label[:,:,0]
    label = tf.cast(label,dtype=tf.uint8)
    label = tf.one_hot(label,depth=2)
    y = label
    return (x_1,x_2),y

# 同形状的交叉熵损失函数CategoricalCrossentropy
# def transform_common(line_x,line_y):
#     image = tf.io.read_file(line_x)
#     image = tf.image.decode_png(image)
#     image = tf.image.resize(image,config.target_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     x = image
#
#     label = tf.io.read_file(line_y)
#     label = tf.image.decode_png(label,channels=1)
#     label = tf.image.resize(label,config.mask_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     label = tf.squeeze(label)
#     label = tf.cast(label,dtype=tf.uint8)
#     label = tf.one_hot(label,depth=2)   # 注意在进行one-hot前 原数组 的书应是【0,1,2,3】  不能是 【0,2,6】 中间不能空
#     label = label[:,:,0]
#     label = tf.cast(label,dtype=tf.uint8)
#     label = tf.one_hot(label,depth=2)
#     y = label
#     return x,y