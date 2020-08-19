# @Time     : 2020/7/19 15:43
# @File     : Dataset
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 数据集父类
# @History  :
#   2020/7/19 Dean First Release
import numpy as np
import os
import tensorflow as tf
from PIL import Image
class Dataset:
    def __init__(self,data_txt_path,target_size,mask_size,num_classes,):
        if os.path.exists(data_txt_path):
            self.data_txt_path = data_txt_path
            self.target_size = target_size
            self.mask_size = mask_size
            self.num_classes = num_classes
        else:
            raise FileNotFoundError("错误,未找到数据集txt文件，{}不存在".format(data_txt_path))
    def getAllDataset(self,seed=7):
        lines_x,lines_y = self.data_txt_to_list(seed)
        dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=(tf.int32,tf.int32),args=(lines_x,lines_y))
        return dataset
    def getTrainValTestDataset(self,rate=(0.8,0.1,0.1),seed=7):
        train,val,test = self.data_txt_to_train_val_test(rate,seed)
        train_dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=(tf.int32,tf.int32),args=train)
        val_dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=(tf.int32,tf.int32),args=val)
        test_dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=(tf.int32,tf.int32),args=test)
        return train_dataset,val_dataset,test_dataset
    def data_txt_to_train_val_test(self,rate,seed):
        lines_x,lines_y = self.data_txt_to_list(seed)
        self.size = len(lines_x)
        i = int(self.size*rate[0])
        j = int(i + self.size*rate[1])
        self.train_size = i
        self.val_size = j
        self.test_size = self.size - j
        lines_x_train,lines_x_val,lines_x_test = lines_x[0:i],lines_x[i:j],lines_x[j:self.size]
        lines_y_train,lines_y_val,lines_y_test = lines_y[0:i],lines_y[i:j],lines_y[j:self.size]
        return (lines_x_train,lines_y_train),(lines_x_val,lines_y_val),(lines_x_test,lines_y_test)
    def data_txt_to_list(self,seed):
        with open(self.data_txt_path,encoding='utf-8') as f:
            lines = f.readlines()
            self.size = len(lines)
        # lines = tf.random.shuffle(lines,seed)
        lines_x = []
        lines_y = []
        for k in lines:
            lines_x.append(os.path.join(os.path.dirname(self.data_txt_path),k.strip().split(';')[0]))
            lines_y.append(os.path.join(os.path.dirname(self.data_txt_path),k.strip().split(';')[1]))
        return lines_x,lines_y

    def getGenerator(self,lines_x,lines_y):
        i=0
        while 1:
            image = Image.open(lines_x[i])
            image = image.resize(self.target_size)
            image_numpy = np.array(image)
            x = image_numpy
            label = Image.open(lines_y[i])
            label = label.resize(self.mask_size)
            label_numpy = np.array(label)
            new_label_numpy = np.zeros(label_numpy.shape + (self.num_classes,)).astype(int)
            for m, n in zip(range(self.num_classes), [0,1]):
                new_label_numpy[:, :, m] = (n == label_numpy[:, :])
            y = new_label_numpy
            if i == (len(lines_x)-1):
                break
            else:
                i=i+1
            yield (x,y)

class CountrySide(Dataset):
    def __init__(self,data_txt_path,target_size,mask_size,num_classes):
        Dataset.__init__(self,data_txt_path,target_size,mask_size,num_classes)
    def getAllDataset(self,seed=7):
        lines_x,lines_y = self.data_txt_to_list(seed)
        dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=((tf.int32,tf.int32),tf.int32),args=(lines_x,lines_y))
        return dataset
    def getTrainValTestDataset(self,rate=(0.8,0.1,0.1),seed=7):
        train,val,test = self.data_txt_to_train_val_test(rate,seed)
        train_dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=((tf.int32,tf.int32),tf.int32),args=train)
        val_dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=((tf.int32,tf.int32),tf.int32),args=val)
        test_dataset = tf.data.Dataset.from_generator(self.getGenerator,output_types=((tf.int32,tf.int32),tf.int32),args=test)
        return train_dataset,val_dataset,test_dataset

    def getGenerator(self,lines_x,lines_y):
        """
        双输入
        """
        i=0
        while 1:
            image = Image.open(lines_x[i])

            image_1 = image.resize(self.target_size)
            image_1_numpy = np.array(image_1)
            x_1 = image_1_numpy

            image_2 = image.resize((3072,3072))
            image_2_numpy = np.array(image_2)
            x_2 = image_2_numpy

            label = Image.open(lines_y[i])

            label = label.resize(self.mask_size)
            label_numpy = np.array(label)
            new_label_numpy = np.zeros(label_numpy.shape + (self.num_classes,)).astype(int)
            for m, n in zip(range(self.num_classes), [0,1]):
                new_label_numpy[:, :, m] = (n == label_numpy[:, :])
            y = new_label_numpy

            if i == (len(lines_x)-1):
                break
            else:
                i=i+1
            yield ((x_1,x_2),y)

if __name__ == '__main__':
    target_size = (512,512)
    mask_size = (64,64)
    num_classes = 2
    batch_size = 4
    dataset = Dataset(r'G:\AI_dataset\dom\segmentation\data.txt',target_size,mask_size,num_classes)
    data,validation_data,test_data = dataset.getTrainValTestDataset()
    for x,y in data:
        y = y[:,:,1]
        for i in range(64):
            for j in range(64):
                print(y[i,j].numpy(),end='')
            print()
        exit()