# @Time     : 2020/7/19 16:30
# @File     : Tools
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 工具脚本
# @History  :
#   2020/7/19 Dean First Release
import requests
import os
import time
import datetime
class Decorator:
    @staticmethod
    def sendEmail(receivers=['1028968939@qq.com'],txt="任务已完成，请抓紧时间处理"):
        def inner(f):
            def inner2(*args,**kwargs):
                f(*args,**kwargs)
                ret = requests.get("https://python.api.dean0731.top/message/sendEmail?receivers={}&txt={}".format(receivers,txt))
                print(ret.content.decode('utf-8'))
                return ret
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
                    print("程序运行{}s".format(end-start))
                    return (ret,end-start)
                else:
                    ret = f(*args,**kwargs)
                return ret
            return inner
        return outer
    @staticmethod
    def sendMessage(data=None):
        def inner(f):
            def inner2(*args,**kwargs):
                ret,seconds = f(*args,**kwargs)
                msg ="任务耗时{}小时{}分钟{}秒,{}".format(*getSecondToTime(seconds),data if data!=None else "")
                url = "https://python.api.dean0731.top/message/sendMessage?content={}".format(msg)
                print(url)
                requests.get(url)
                return ret
            return inner2
        return inner

def sendMessage(data=None):
    """
    向企业我的微信发送信息
    """
    print(data)
    if data == None:
        ret = requests.get("https://python.api.dean0731.top/message/sendMessage")
    else:
        ret = requests.get("https://python.api.dean0731.top/message/sendMessage?content={}".format(data))
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

def get_dir(parent=None):
    """ 生成目录
    工具脚本，生成文件目录，
    ~/log-data-time
        -event
        -h5
    """
    time = datetime.datetime.now()
    if parent != None:
        parent = os.path.join(parent,'source')
    else:
        parent = 'source'
    log_dir = os.path.join(parent,'logs-{}-{}-{}-{}-{}-{}'.format(time.year,time.month,time.day,time.hour,time.minute,time.second))
    # log_dir = os.path.join('source','logs')
    h5_dir = os.path.join(log_dir,'h5')
    event_dir = os.path.join(log_dir,'event')
    for i in [log_dir,h5_dir,event_dir]:
        if not os.path.exists(i):
            os.makedirs(i)
    return log_dir,h5_dir,event_dir

def getNumbySize(num,n):
    if type(num) == 'float':
        return round(num,n)
    else:
        return [ round(i,n) for i in num]
def getSecondToTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return int(h),int(m),round(s,2)

if __name__ == '__main__':
    from tf.network import Model

    # model = Model.getModel('mysegnet',(2816,2816),n_labels=2)
    target_size = 256,256
    model = Model.getModel('mysegnet_3', target_size, n_labels=2)
    # model.summary()
    computerNetworkSize(model)
