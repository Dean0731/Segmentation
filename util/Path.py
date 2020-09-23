# @Time     : 2020/8/25 18:02
# @File     : DatasetPath
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import getpass
import socket
class DatasetPath:
    def __init__(self,dataset='dom'):
        if dataset.lower() == 'dom':
            self.__getDom()
        elif dataset.lower() == 'fashion_mnist':
            self.__getFashionMinst()
        elif dataset.lower() == 'mnist':
            self.__getMinst()
        else:
            raise FileNotFoundError("数据集不存在")
    def __getDom(self):
        self.Shiyanshi = r'E:\DeepLearning\AI_dataset\dom\png_png\data.txt'
        self.P2000= r'/home/dean/PythonWorkSpace/Segmentation/dataset/dom/segmentation/data.txt'
        self.lENOVO_PC = r'G:\AI_dataset\dom\segmentation\data.txt'
        self.TI1050 = '/home/dean/dataset/dom/png_png/data.txt'
        self.Chaosuan = r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation/data.txt'
        self.Aistudio = r'/home/aistudio/work/dataset/dom/data.txt'
        self.Aliyun = r'/home/admin/jupyter/dataset/dom/data.txt'
        self.Huawei = r'/home/ma-user/work/dataset/dom/segmentation\data.txt'
    def __getMinst(self):
        self.Shiyanshi = r'E:\Workspace\PythonWorkSpace\Segmentation\dataset\MNIST'
        self.P2000= r''
        self.lENOVO_PC = r'G:\AI_dataset\MNIST'
        self.Chaosuan = r''
        self.Aistudio = r'/home/aistudio/work/dataset/MNIST'
        self.Aliyun = r'/home/admin/jupyter/dataset/MNIST'
        self.Huawei = r'/home/ma-user/work/dataset/MNIST'
    def __getFashionMinst(self):
        self.Shiyanshi = r''
        self.P2000= r''
        self.lENOVO_PC = r'G:\AI_dataset\fashion-mnist'
        self.Chaosuan = r''
        self.Aistudio = r'/home/aistudio/work/dataset/fashion-mnist'
        self.Aliyun = r'/home/admin/jupyter/dataset/fashion-mnist'
        self.Huawei = r'/home/ma-user/work/dataset/fashion-mnist'
    def getPath(self):
        user_name = getpass.getuser()
        host_name = socket.gethostname();
        if user_name == 'aistudio':
            return self.Aistudio
        elif user_name == 'Administrator':
            return self.lENOVO_PC
        elif user_name == 'dean' and host_name == 'P2000':
            return self.P2000
        elif user_name =='admin':
            return self.Aliyun
        elif user_name =='root':
            return self.Shiyanshi
        elif user_name =='weiht':
            return self.Chaosuan
        elif user_name == 'ma-user':
            return self.Huawei
        elif host_name == 'dean-1050TI':
            return self.TI1050
        else:
            raise FileNotFoundError("根据环境选数据集位置失败")