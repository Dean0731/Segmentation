# @Time     : 2020/7/19 15:43
# @File     : Dataset
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 数据集父类
# @History  :
#   2020/7/19 Dean First Release
import numpy as np
import logging
import os
from tensorflow import keras
from util import Tools
from util.dataset.AerialImage import AerialImage
from util.dataset.CountrySide import CountrySide
from util.dataset.Massachusetts import Massachusetts

class Dataset():
    def __init__(self,parent,dir=('images', 'gt'),shapeToOneDimension = False,data_size='tif_576'):
        self.train_dir = ''
        self.val_dir = ''
        self.test_dir = ''
        self.parent = parent
        self.img_name, self.label_name = dir
        self.shapeToOneDimension = shapeToOneDimension
        self.setDataset(data_size)
        self.data_gen_args = dict(featurewise_center=False,
                             samplewise_center=False, featurewise_std_normalization=False,
                             samplewise_std_normalization=False, zca_whitening=False,
                             zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0,
                             height_shift_range=0.0, brightness_range=None, shear_range=0.0,
                             zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest',
                             cval=0.0, horizontal_flip=False, vertical_flip=False,
                             rescale=None, preprocessing_function=None, data_format=None,
                             validation_split=0.0)
        self.size = {'train':[],'val':[],'testNetwork':[]}
    def setDataset(self,flag='tif_576'):
        pass
    def __getDir(self,data_type):
        if data_type == 'train':
            data_type = self.train_dir
        elif data_type =="val":
            data_type = self.val_dir
        elif data_type =="testNetwork":
            data_type = self.test_dir
        else:
            logging.error("数据错误,没有{}参数".format(data_type))
            exit()
        if self.parent != '':
            data_type = os.path.join(self.parent,data_type)
        return data_type
    def __getGenerator(self,target_size,mask_size,batch_size,data_type,seed=7,num_classes=2,flag=False):
        data_type = self.__getDir(data_type)
        image_datagen = keras.preprocessing.image.ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(data_type, target_size=target_size, classes=[self.img_name],
                                                            class_mode=None, seed=seed, batch_size=batch_size,
                                                            color_mode='rgb')
        label_generator = image_datagen.flow_from_directory(data_type, target_size=mask_size, classes=[self.label_name],
                                                            class_mode=None, seed=seed, batch_size=batch_size,
                                                            color_mode='grayscale')
        train_generator = zip(image_generator, label_generator)
        for (img, mask) in train_generator:
            img, mask = self.__adjustData(img, mask, num_classes,flag)
            yield (img, mask)

    def __adjustData(self,img, mask, num_classes,flag):
        #img = img / 255
        mask = mask[:, :, :, 0].astype(int) if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_classes,)).astype(int)  # (2, 416, 416, 2)
        for i, j in zip(range(num_classes), [0, 255]):
            new_mask[:, :, :, i] = (j == mask[:, :, :])
        if flag:
            mask = new_mask.reshape((-1,new_mask.shape[1]*new_mask.shape[2],num_classes))
        else:
            mask = new_mask
        return (img, mask)

    def getData(self,target_size=(64, 64),mask_size=(64,64),batch_size = 4):
        data = self.__getGenerator(target_size=target_size, mask_size=mask_size, batch_size=batch_size, data_type='train', flag=False)
        validation_data = self.__getGenerator(target_size, mask_size=mask_size, batch_size=batch_size, data_type='val', flag=False)
        test_data = self.__getGenerator(target_size=target_size, mask_size=mask_size, batch_size=batch_size, data_type='testNetwork', flag=False)
        return data,validation_data,test_data
    def getSize(self,data_type):
        if self.size[data_type] == []:
            self.size[data_type] = Tools.countNumOfFolder(self.__getDir(data_type))
        return int(self.size.get(data_type)[0]/2)
def selectDataset(str='A',data_size='tif_576'):
    if str == 'A':
        dataset = AerialImage(parent='G:\AI_dataset\法国-房屋数据集2',data_size=data_size)
    elif str == 'C':
        dataset = CountrySide(parent='G:\AI_dataset\DOM',data_size=data_size)
    elif str == 'M':
        dataset = Massachusetts(parent='G:\AI_dataset\马萨诸塞州-房屋数据集1',data_size=data_size)
    else:
        print("错误:未找到数据集.")
    return dataset