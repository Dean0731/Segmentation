from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
##这一步获取了原图和mask
def trainGenerator(batch_size,train_path,train_path2,image_folder,mask_folder,aug_dict,
                   flag_multi_class, num_class ,image_color_mode = "grayscale",
                   mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",save_to_dir = None,target_size = (256,256),seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path2,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,
                              flag_multi_class,   ##参数（flag_multi_class）用来开启多分类
                              num_class)##这里对图像进行了处理，用以减小计算量（函数定义在下面）
        yield (img,mask)##用生成器进行迭代数据，可以传入model.fit_generator（）这个函数进行训练

def adjustData(img,mask,flag_multi_class,num_class):
    if (flag_multi_class):#如果多分类，在mask添加多层，每层对应一个类别
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.ones(mask.shape + (len(num_class),))
        for i in range(len(num_class)):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == num_class[i], i] = 0
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1], new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0], new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):#如果不是多分类，直接对img，mask进行操作，不难理解
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)
