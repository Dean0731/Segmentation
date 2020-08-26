# @Time     : 2020/7/19 15:43
# @File     : Dataset
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 数据集读取
# @History  :
#   2020/7/19 Dean First Release
import numpy as np
np.set_printoptions(threshold = 1e6)
import os
import numpy as np
import tensorflow as tf
class Dataset:
    """
    单输入数据读取
    """
    def __init__(self,data_txt_path,target_size,mask_size,num_classes):
        if os.path.exists(data_txt_path):
            self.data_txt_path = data_txt_path
            self.target_size = target_size
            self.mask_size = mask_size
            self.num_classes = num_classes
        else:
            raise FileNotFoundError("错误,未找到数据集txt文件，{}不存在".format(data_txt_path))
    def getAllDataset(self,seed=None):
        lines_x,lines_y = self.data_txt_to_list(seed)
        dataset = tf.data.Dataset.from_tensor_slices((lines_x,lines_y)).map(self.getGenerator)
        return dataset
    def getTrainValTestDataset(self,rate=(0.8,0.1,0.1),seed=None):
        dataset = self.getAllDataset(seed)
        train = int(len(dataset)*rate[0])
        val = int(len(dataset)*rate[1])
        train_dataset = dataset.take(train)
        val_dataset = dataset.skip(train).take(val)
        test_dataset = dataset.skip(train+val)
        return train_dataset,val_dataset,test_dataset

    def data_txt_to_list(self,seed):
        with open(self.data_txt_path,encoding='utf-8') as f:
            lines = f.readlines()
            self.size = len(lines)
        if seed !=None:
            np.random.seed(seed)
            np.random.shuffle(lines)
        lines_x = []
        lines_y = []
        for k in lines:
            lines_x.append(os.path.join(os.path.dirname(self.data_txt_path),k.strip().split(';')[0]))
            lines_y.append(os.path.join(os.path.dirname(self.data_txt_path),k.strip().split(';')[1]))
        return lines_x,lines_y

    def getGenerator(self,line_x,line_y):
        image = tf.io.read_file(line_x)
        image = tf.image.decode_png(image)
        image = tf.image.resize(image,self.target_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = image

        label = tf.io.read_file(line_y)
        label = tf.image.decode_png(label,channels=1)
        label = tf.image.resize(label,self.mask_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = tf.squeeze(label)
        label = tf.cast(label,dtype=tf.uint8)
        label = tf.one_hot(label,depth=2)   # 注意在进行one-hot前 原数组 的书应是【0,1,2,3】  不能是 【0,2,6】 中间不能空
        label = label[:,:,0]
        label = tf.cast(label,dtype=tf.uint8)
        label = tf.one_hot(label,depth=2)
        y = label
        return x,y

class CountrySide(Dataset):
    """
    双输入
    """
    def __init__(self,data_txt_path,target_size,mask_size,num_classes):
        Dataset.__init__(self,data_txt_path,target_size,mask_size,num_classes)
    def getGenerator(self,line_x,line_y):
        image = tf.io.read_file(line_x)
        image = tf.image.decode_png(image)
        image_1 = tf.image.resize(image,self.target_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_2 = tf.image.resize(image,(3072,3072),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x_1,x_2 = image_1,image_2

        label = tf.io.read_file(line_y)
        label = tf.image.decode_png(label,channels=1)
        label = tf.image.resize(label,self.mask_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = tf.squeeze(label)
        label = tf.cast(label,dtype=tf.uint8)
        label = tf.one_hot(label,depth=2)
        label = label[:,:,0]
        label = tf.cast(label,dtype=tf.uint8)
        label = tf.one_hot(label,depth=2)
        y = label
        return (x_1,x_2),y

if __name__ == '__main__':
    target_size = (512,512)
    mask_size = (64,64)
    num_classes = 2
    # dataset = CountrySide(r'G:\AI_dataset\dom\segmentation3\data.txt',target_size,mask_size,num_classes)
    # data,test,val = dataset.getTrainValTestDataset()
    dataset = Dataset(r'G:\AI_dataset\dom\segmentation3\data.txt',target_size,mask_size,num_classes)
    data = dataset.getAllDataset()
    # data = data.batch(batch_size)
    for x,y in data:
        # x_1,x_2 = x
        # print(x_1.shape)
        # print(x_2.shape)
        print(x.shape)
        print(y.shape)
        y = y[:,:,0]
        print(y.shape)
        for i in range(64):
            for j in range(64):
                print(int(y[i,j].numpy()),end='')
            print()
        exit()