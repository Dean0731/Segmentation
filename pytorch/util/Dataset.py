# @Time     : 2020/8/25 17:57
# @File     : Dataset
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import os
from PIL import Image
from torch.utils.data import Dataset
import util
class Dataset(Dataset):
    # 初始化读取txt 可以设定变换
    def __init__(self, data_txt_path,transform = None, target_transform = None):
        if os.path.exists(data_txt_path):
            self.data_txt_path = data_txt_path
            self.transform = transform
            self.target_transform = target_transform
            x,y = util.data_txt_to_list(self.data_txt_path,seed=None,split=';')
            self.imgs = list(zip(x,y))
        else:
            raise FileNotFoundError("错误,未找到数据集txt文件，{}不存在".format(data_txt_path))

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