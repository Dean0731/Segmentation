# @Time     : 2020/8/25 18:02
# @File     : DatasetPath
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release

import time
import logging
import os,sys
from .func import getSecondToTime,getUrlAndLog,get_name_hostname
class DatasetPath:
    TRAIN = 0
    VAL = 1
    TEST = 2
    ALL = 3
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
        self.Shiyanshi = r'E:\DeepLearning\AI_dataset\dom\png_png'
        self.P2000= r'/home/dean/dataset/dom/png_png'
        self.lENOVO_PC = r'G:\AI_dataset\dom\segmentation'
        self.TI1050 = '/home/dean/dataset/dom/png_png'
        self.Chaosuan = r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation'
        self.Aistudio = r'/home/aistudio/dataset/dom'
        self.Aliyun = r'/home/admin/jupyter/dataset/dom'
        self.Huawei = r'/home/ma-user/work/dataset/dom/segmentation'
    def __getMinst(self):
        self.Shiyanshi = r'E:\Workspace\PythonWorkSpace\Segmentation\dataset\MNIST'
        self.P2000= r''
        self.lENOVO_PC = r'G:\AI_dataset\MNIST'
        self.Chaosuan = r''
        self.Aistudio = r'/home/aistudio/dataset/MNIST'
        self.Aliyun = r'/home/admin/jupyter/dataset/MNIST'
        self.Huawei = r'/home/ma-user/work/dataset/MNIST'
    def __getFashionMinst(self):
        self.Shiyanshi = r''
        self.P2000= r''
        self.lENOVO_PC = r'G:\AI_dataset\fashion-mnist'
        self.Chaosuan = r''
        self.Aistudio = r'/home/aistudio/dataset/fashion-mnist'
        self.Aliyun = r'/home/admin/jupyter/dataset/fashion-mnist'
        self.Huawei = r'/home/ma-user/work/dataset/fashion-mnist'
    def getPath(self,type):
        user_name,host_name = get_name_hostname()
        if type == DatasetPath.TRAIN:
            file = 'data_train.txt'
        elif type == DatasetPath.VAL:
            file = 'data_val.txt'
        elif type == DatasetPath.TEST:
            file = 'data_test.txt'
        else:
            file = 'data.txt'
        if user_name == 'aistudio':
            parent = self.Aistudio
        elif user_name == 'Administrator':
            parent = self.lENOVO_PC
        elif user_name == 'dean' and host_name == 'P2000':
            parent = self.P2000
        elif user_name == 'dean' and host_name == '2070':
            parent = self.P2000
        elif user_name =='admin':
            parent = self.Aliyun
        elif user_name =='root':
            parent =  self.Shiyanshi
        elif user_name =='weiht':
            parent = self.Chaosuan
        elif user_name == 'ma-user':
            parent = self.Huawei
        elif host_name == 'dean-1050TI':
            parent = self.TI1050
        else:
            raise FileNotFoundError("根据环境选数据集位置失败")
        return os.path.join(parent,file)
class Decorator:
    @staticmethod
    def _messageHandler(seconds,message):
        return "任务耗时{}小时{}分钟{}秒,{}".format(*getSecondToTime(seconds),message)
    @staticmethod
    def sendEmail(receivers='1028968939@qq.com',message="任务已完成，请抓紧时间处理"):
        def inner(f):
            def inner2(*args,**kwargs):
                ret,seconds = f(*args,**kwargs)
                msg = Decorator._messageHandler(seconds,message)
                url = "https://python.api.dean0731.top/message/sendEmail?receivers={}&txt={}".format(receivers,msg)
                return getUrlAndLog(url)
            return inner2
        return inner
    @staticmethod
    def timer(flag=True):
        def outer(f):
            def inner(*args,**kwargs):
                if flag:
                    start = time.time()
                    ret = f(*args,**kwargs)
                    time.sleep(1)
                    end = time.time()
                    logging.info("程序运行{}s".format(end-start))
                    return (ret,end-start)
                else:
                    ret = f(*args,**kwargs)
                return ret
            return inner
        return outer
    @staticmethod
    def sendMessageWeChat(message=''):
        def inner(f):
            def inner2(*args,**kwargs):
                ret,seconds = f(*args,**kwargs)
                msg = Decorator._messageHandler(seconds,message)
                url = "https://python.api.dean0731.top/message/sendMessageWeChat?content={}".format(msg)
                return getUrlAndLog(url)
            return inner2
        return inner
    @staticmethod
    def sendMessageDingTalk(message=None,tels:str=None,all:str=False):
        def inner(f):
            def inner2(*args,**kwargs):
                ret,seconds = f(*args,**kwargs)
                msg = Decorator._messageHandler(seconds,message)
                url = "https://python.api.dean0731.top//message/sendMessageToDingTalk?message={}&tels={}&all={}".format(msg,tels,all)
                return getUrlAndLog(url)
            return inner2
        return inner
class progressbar(object):

    def __init__(self, count,block_char='.'):
        self.block = block_char
        self.f = sys.stdout
        self.scale = count/100
        self.last = 0
    def progress(self, i):
        if(i - self.last) * self.scale >=self.scale:
            print()

        if self.finalcount:
            percentcomplete = int(round(100.0 * count / self.finalcount))
            if percentcomplete < 1:
                percentcomplete = 1
        else:
            percentcomplete = 100
        blockcount = int(percentcomplete // 2)
        if blockcount <= self.blockcount:
            return
        for i in range(self.blockcount, blockcount):
            self.f.write(self.block)
        self.f.flush()
        self.blockcount = blockcount
        if percentcomplete == 100:
            self.f.write("\n")