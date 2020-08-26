# @Time     : 2020/8/25 17:57
# @File     : Dataset
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
class Dataset(Dataset):
    # 初始化读取txt 可以设定变换
    def __init__(self, data_txt_path, type,transform = None, target_transform = None):
        if os.path.exists(data_txt_path):
            self.data_txt_path = data_txt_path
            self.type = type
            self.transform = transform
            self.target_transform = target_transform
            self.imgs = list(zip(*self.data_txt_to_list(self.__getLines())))
        else:
            raise FileNotFoundError("错误,未找到数据集txt文件，{}不存在".format(data_txt_path))
    def __getLines(self):
        with open(self.data_txt_path,encoding='utf-8') as f:
            lines = f.readlines()
        a = int(len(lines)*0.8)
        b = int(len(lines)*0.9)
        if self.type == 'train':
            lines = lines[0:a]
        elif self.type == 'val':
            lines = lines[a:b]
        elif self.type == 'test':
            lines = lines[b:]
        else:
            raise ValueError("未找到{}数据集".format(self.type))
        return lines
    def data_txt_to_list(self,lines):
        lines_x = []
        lines_y = []
        for k in lines:
            lines_x.append(os.path.join(os.path.dirname(self.data_txt_path),k.strip().split(';')[0]))
            lines_y.append(os.path.join(os.path.dirname(self.data_txt_path),k.strip().split(';')[1]))
        return lines_x,lines_y
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn)
        if self.transform is not None:
            for i in self.transform:
                img = i(img)
        label = Image.open(label)
        if self.target_transform is not None:
            for i in self.target_transform:
                label = i(label)
        return img, label
    def __len__(self):
        return len(self.imgs)