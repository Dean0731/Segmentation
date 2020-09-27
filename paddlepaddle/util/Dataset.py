# @Time     : 2020/9/27 11:22
# @File     : util
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/9/27 Dean First Release
#导入要用到的模块
import paddle
import os
from multiprocessing import cpu_count
class Dataset:
    """
    单输入数据读取
    """
    def __init__(self,data_txt_path):
        if os.path.exists(data_txt_path):
            self.data_txt_path = data_txt_path
        else:
            raise FileNotFoundError("错误,未找到数据集txt文件，{}不存在".format(data_txt_path))
    def getDataset(self,transform,buffered_size=1024):
        def reader():
            with open(self.data_txt_path, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    # 图像的路径和标签是以\t来分割的,所以我们在生成这个列表的时候,使用\t就可以了
                    img_path, lab = line.strip().split(';')
                    yield os.path.join(os.path.dirname(self.data_txt_path),img_path), os.path.join(os.path.dirname(self.data_txt_path),lab)
                    # 创建自定义数据训练集的train_reader
        return paddle.reader.xmap_readers(transform, reader, cpu_count(), buffered_size)