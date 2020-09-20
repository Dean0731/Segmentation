# @Time     : 2020/7/19 15:43
# @File     : Dataset
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 数据集读取
# @History  :
#   2020/7/19 Dean First Release
import os
import tensorflow as tf
from util import Tools
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
    def getDataset(self,transform,seed=7,split=(0.8,0.1,0.1)):
        lines_x,lines_y = Tools.data_txt_to_list(self.data_txt_path,seed)
        dataset_sum = len(lines_x)
        dataset = tf.data.Dataset.from_tensor_slices((lines_x,lines_y)).map(transform)
        if split == None:
            return dataset
        else:
            train = int(dataset_sum*split[0])
            val = int(dataset_sum*split[1])
            train_dataset = dataset.take(train)
            val_dataset = dataset.skip(train).take(val)
            test_dataset = dataset.skip(train+val)
            return train_dataset,val_dataset,test_dataset

if __name__ == '__main__':
    pass