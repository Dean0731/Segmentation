import time
import requests


def sendEmail(receivers=['1028968939@qq.com'],txt="任务已完成，请抓紧时间处理"):
    def inner(f):
        def inner2(*args,**kwargs):
            f(*args,**kwargs)
            ret = requests.get("https://python.api.dean0731.top/message/sendEmail?receivers={}&txt={}".format(receivers,txt))
            print(ret.content.decode('utf-8'))
            return ret
        return inner2
    return inner


def timer(flag=False):
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


def sendMessage(data=None):
    def inner(f):
        def inner2(*args,**kwargs):
            f(*args,**kwargs)
            if data == None:
                ret = requests.get("https://python.api.dean0731.top/message/sendMessage")
            else:
                ret = requests.get("https://python.api.dean0731.top/message/sendMessage?content={}".format(data))
            print(ret.content.decode('utf-8'))
            return ret
        return inner2
    return inner

if __name__ == "__main__":
    pass
