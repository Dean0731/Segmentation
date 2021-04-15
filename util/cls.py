# @Time     : 2020/8/25 18:02
# @File     : DatasetPath
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release

import time
import logging
import os
from .func import getSecondToTime, getUrlAndLog, get_name_hostname

class DatasetPath:
    TRAIN = 0
    VAL = 1
    TEST = 2
    ALL = 3
    Path = {
        "dom": {
            "root": r"/public/home/huzhanpeng/Dataset/dom/png_png"
        },
        "fashion_mnist": {

        },
        "mnist": {

        }
    }

    def __init__(self, dataset='dom'):
        self.dataset = dataset

    def getPath(self, type):
        user_name, host_name = get_name_hostname()
        map = {
            DatasetPath.TRAIN: 'data_train.txt',
            DatasetPath.VAL: 'data_val.txt',
            DatasetPath.TEST: 'data_test.txt',
            DatasetPath.ALL: 'data.txt',
        }
        parent = DatasetPath.Path[self.dataset][user_name]
        file = map[type]
        assert file is not None
        assert parent is not None
        return os.path.join(parent, file)

    def __str__(self):
        return self.getPath(DatasetPath.TRAIN)


class Decorator:
    @staticmethod
    def _messageHandler(seconds, message):
        return "任务耗时{}小时{}分钟{}秒,{}".format(*getSecondToTime(seconds), message)

    @staticmethod
    def sendEmail(receivers='1028968939@qq.com', message="任务已完成，请抓紧时间处理"):
        def inner(f):
            def inner2(*args, **kwargs):
                ret, seconds = f(*args, **kwargs)
                msg = Decorator._messageHandler(seconds, message)
                url = "http://dean0731.top/api/util/message/sendEmail?receivers={}&txt={}".format(receivers, msg)
                return getUrlAndLog(url)

            return inner2

        return inner

    @staticmethod
    def timer(flag=True):
        def outer(f):
            def inner(*args, **kwargs):
                if flag:
                    start = time.time()
                    ret = f(*args, **kwargs)
                    time.sleep(1)
                    end = time.time()
                    logging.info("程序运行{}s".format(end - start))
                    return (ret, end - start)
                else:
                    ret = f(*args, **kwargs)
                return ret

            return inner

        return outer

    @staticmethod
    def sendMessageWeChat(message='', flag=True):
        def inner(f):
            def inner2(*args, **kwargs):
                if flag:
                    ret, seconds = f(*args, **kwargs)
                    msg = Decorator._messageHandler(seconds, message)
                    url = "http://dean0731.top/api/util/message/sendMessageWeChat?content={}".format(msg)
                    return getUrlAndLog(url)
                else:
                    ret, seconds = f(*args, **kwargs)

            return inner2

        return inner

    @staticmethod
    def sendMessageDingTalk(message=None, tels: str = None, all: str = False, flag=True):
        def inner(f):
            def inner2(*args, **kwargs):
                if flag:
                    ret, seconds = f(*args, **kwargs)
                    msg = Decorator._messageHandler(seconds, message)
                    url = "http://dean0731.top/api/util/message/sendMessageToDingTalk?message={}&tels={}&all={}".format(
                        msg, tels, all)
                    return getUrlAndLog(url)
                else:
                    ret, seconds = f(*args, **kwargs)

            return inner2

        return inner
