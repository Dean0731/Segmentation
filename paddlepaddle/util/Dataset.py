# @Time     : 2020/9/27 11:22
# @File     : util
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/9/27 Dean First Release
import os
from paddle.io import Dataset
import util
class Dataset(Dataset):
    """
    数据集定义
    """
    def __init__(self, data_txt_path,transform,seed=7):
        """
        构造函数
        """
        if os.path.exists(data_txt_path):
            self.data_txt_path = data_txt_path
        else:
            raise FileNotFoundError("错误,未找到数据集txt文件，{}不存在".format(data_txt_path))
        self.transform = transform
        self.line_x,self.line_y = util.data_txt_to_list(data_txt_path,seed)
    def __getitem__(self, idx):
        """
        返回 image, label
        """
        x = self.transform(self.line_x[idx],mode='image')
        y = self.transform(self.line_y[idx],mode='label')
        return x, y.astype('int64')

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.line_x)
    def getDataset(self,split=(0.8,0.1,0.1)):
        num = len(self)
        a = int(num*split[0])
        b = int(num*split[1])
        return self[0:a],self[a:a+b],self[a+b:]
