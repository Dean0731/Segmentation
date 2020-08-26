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
                print("未找到到数据集")
                exit()
    def __getDom(self):
        self.Shiyanshi_benji = r'E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation2\data.txt'
        self.Shiyanshi_hu= r'/home/dean/PythonWorkSpace/Segmentation/dataset/dom/segmentation2/data.txt'
        self.lENOVO_PC = r'G:\AI_dataset\dom\segmentation2\data.txt'
        self.Chaosuan = r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation2/data.txt'
        self.Aistudio = r'/home/aistudio/work/dataset/dom/data.txt'

    def __getMinst(self):
        self.lENOVO_PC = r'G:\AI_dataset\MNIST'
        self.Aistudio = r'/home/aistudio/work/dataset/MNIST'
    def getFashionMinst(self):
        self.lENOVO_PC = r'G:\AI_dataset\fashion-mnist'
        self.Aistudio = r'/home/aistudio/work/dataset/fashion-mnist'
    def getPath(self):
        user_name = getpass.getuser()
        if user_name == 'aistudio':
            return self.Aistudio
        elif user_name == 'Administrator':
            return self.lENOVO_PC
        elif user_name == 'dean':
            return self.Shiyanshi_hu
        else:
            return self.Shiyanshi_benji