# @Time     : 2020/7/19 16:30
# @File     : Tools
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 工具脚本
# @History  :
#   2020/7/19 Dean First Release
import requests
import os,numpy as np
import sys
import logging
import cv2
import getpass
import socket
import datetime
from PIL import Image
def get_name_hostname():
    user_name = getpass.getuser()
    host_name = socket.gethostname()
    return user_name,host_name
def get_dir(parent=None):
    """ 生成目录
    工具脚本，生成文件目录，
    ~/log-data-time
        -event
        -h5
    """
    time = datetime.datetime.now()
    if parent == None:
        parent = 'source'
    log_dir = os.path.join(parent,'logs-{}-{}-{}-{}-{}-{}'.format(time.year,time.month,time.day,time.hour,time.minute,time.second))
    # log_dir = os.path.join('source','logs')
    h5_dir = os.path.join(log_dir,'h5')
    event_dir = os.path.join(log_dir,'event')
    for i in [log_dir,h5_dir,event_dir]:
        if not os.path.exists(i):
            os.makedirs(i)
    return log_dir,h5_dir,event_dir
def sendEmail(receivers='1028968939@qq.com',txt="任务已完成，请抓紧时间处理"):
    """
    发邮件
    """
    url = "https://python.api.dean0731.top/message/sendEmail?receivers={}&txt={}".format(receivers,txt);
    return getUrlAndLog(url)

def sendMessageDingTalk(message='',tels:str=None,all:str=False):
    """
    向钉钉发送消息
    """
    url = "https://python.api.dean0731.top//message/sendMessageDingTalk?message={}&tels={}&all={}".format(message,tels,all)
    return getUrlAndLog(url)
def sendMessageWeChat(message=''):
    """
    向企业我的微信发送信息
    """
    url = "https://python.api.dean0731.top/message/sendMessageWeChat?content={}".format(message)
    return getUrlAndLog(url)

def getUrlAndLog(url):
    ret = requests.get(url).content.decode('utf-8')
    logging.info(ret)
    return ret

def countNumOfFolder(path):
    file = 0
    folder = 0
    for root, dirs, files in os.walk(path, topdown=False):
        file = file + len(files)
        folder = folder + len(dirs)
    return [file,folder]

def computerNetworkSize(model):
    # from keras import applications
    # model = applications.VGG16(input_shape=(576,576,3),include_top=False,weights=None)
    # print("无全连接层总参数量:",model.count_params())
    # model = applications.VGG16(input_shape=(576,576,3),include_top=True,weights=None)
    # print("有全连接层总参数量:",model.count_params())
    print("总参数量:",model.count_params())
    all_params_memory = 0
    all_feature_memory = 0
    for num,layer in enumerate(model.layers):
        #训练权重w占用的内存
        params_memory = layer.count_params()*4/(1024*1024)
        all_params_memory = all_params_memory + params_memory
        #特征图占用内存
        feature_shape = layer.output_shape
        if type(feature_shape) is list :
            feature_shape = feature_shape[0]
        feature_size=1
        for i in range(1,len(feature_shape)):
            feature_size = feature_size*feature_shape[i]
        feature_memory = feature_size*4/(1024*1024)
        print("layer:{}".format(num).ljust(10,' '),
              "特征图占用内存:{}".format(feature_shape).ljust(33,' '),"{}".format(feature_size).ljust(10,' '),"{}M".format(str(feature_memory)).ljust(10,' '),
              "训练权重w占用的内存:{}".format(layer.name).ljust(33,' '),"{}".format(layer.count_params()).ljust(10,' '),"{}M".format(str(params_memory))
              )
        all_feature_memory = all_feature_memory + feature_memory
    print()
    print("网络权重W占用总内存:",str(all_params_memory)+"M","网络特征图占用总内存:",str(all_feature_memory)+"M")
    print("网络总消耗内存:",str(all_params_memory+all_feature_memory)+"M")



def getNumbySize(num,n):
    if type(num) == 'float':
        return round(num,n)
    else:
        return [ round(i,n) for i in num]
def getSecondToTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return int(h),int(m),round(s,2)
def getCmdDict():
    argv = sys.argv
    keys = argv[1:-1:2]
    for i,key in enumerate(keys):
        if key.startswith('--'):
            keys[i] = key[2:len(key)]
        else:
            raise Exception("参数格式不正确:{}".format(key))
    values = argv[2:len(argv):2]
    return dict(zip(keys,values))
def data_txt_to_list(data_txt_path,seed=None,split=None):
    if split == None:
        name,host = get_name_hostname()
        if name == 'aistudio':
            split = ' '
        else:
            split = ';'
    with open(data_txt_path,encoding='utf-8') as f:
        lines = f.readlines()
    if seed !=None:
        np.random.seed(seed)
        np.random.shuffle(lines)
        np.random.seed(seed-1)
        np.random.shuffle(lines)
    lines_x = []
    lines_y = []
    for k in lines:
        lines_x.append(os.path.join(os.path.dirname(data_txt_path),k.strip().split(split)[0]))
        lines_y.append(os.path.join(os.path.dirname(data_txt_path),k.strip().split(split)[1]))
    return lines_x,lines_y
def printImagePIL(img_path,all=False):
    img = Image.open(img_path)
    print("image size:{}".format(img.size))
    print("image mode:{}".format(img.mode))
    if not all:
        img = img.resize((64,64),resample=0)
    return np.array(img)
def printArray(array:np.array):
    print("打印像素值：",np.unique(array))
    shape = array.shape
    if len(shape) == 1:
        for i in range(shape[0]):
            print(array[i],end='')
    elif len(shape) >= 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                print(array[i,j],end='')
            print()
def printImagecv2(img_path,all=False):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    print("image size:{}".format(img.shape))
    if not all:
        img = cv2.resize(img,(64,64),interpolation=cv2.INTER_NEAREST)
    return np.array(img)
if __name__ == '__main__':
    from tf.network import Model

    # model = Model.getModel('mysegnet',(2816,2816),n_labels=2)
    target_size = 256,256
    model = Model.getModel('mysegnet_3', target_size, n_labels=2)
    # model.summary()
    computerNetworkSize(model)
