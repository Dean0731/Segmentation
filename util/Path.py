# @Time     : 2020/8/25 18:02
# @File     : DatasetPath
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import getpass
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
        self.Shiyanshi_benji = r'E:\DeepLearning\AI_dataset\dom\png_png\data.txt'
        self.Shiyanshi_hu= r'/home/dean/PythonWorkSpace/Segmentation/dataset/dom/segmentation/data.txt'
        self.lENOVO_PC = r'G:\AI_dataset\dom\segmentation\data.txt'
        self.Chaosuan = r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation/data.txt'
        self.Aistudio = r'/home/aistudio/work/dataset/dom/data.txt'
        self.Aliyun = r'/home/admin/jupyter/dataset/dom/data.txt'
        self.Huawei = r'/home/ma-user/work/dataset/dom/segmentation\data.txt'
    def __getMinst(self):
        self.Shiyanshi_benji = r'E:\Workspace\PythonWorkSpace\Segmentation\dataset\MNIST'
        self.Shiyanshi_hu= r''
        self.lENOVO_PC = r'G:\AI_dataset\MNIST'
        self.Chaosuan = r''
        self.Aistudio = r'/home/aistudio/work/dataset/MNIST'
        self.Aliyun = r'/home/admin/jupyter/dataset/MNIST'
        self.Huawei = r'/home/ma-user/work/dataset/MNIST'
    def __getFashionMinst(self):
        self.Shiyanshi_benji = r''
        self.Shiyanshi_hu= r''
        self.lENOVO_PC = r'G:\AI_dataset\fashion-mnist'
        self.Chaosuan = r''
        self.Aistudio = r'/home/aistudio/work/dataset/fashion-mnist'
        self.Aliyun = r'/home/admin/jupyter/dataset/fashion-mnist'
        self.Huawei = r'/home/ma-user/work/dataset/fashion-mnist'
    def getPath(self):
        user_name = getpass.getuser()
        if user_name == 'aistudio':
            return self.Aistudio
        elif user_name == 'Administrator':
            return self.lENOVO_PC
        elif user_name == 'dean':
            return self.Shiyanshi_hu
        elif user_name =='admin':
            return self.Aliyun
        elif user_name =='root':
            return self.Shiyanshi_benji
        elif user_name =='weiht':
            return self.Chaosuan
        elif user_name == 'ma-user':
            return self.Huawei
        else:
            raise FileNotFoundError("根据环境选数据集位置失败")