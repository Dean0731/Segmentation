# @Time     : 2020/7/19 16:30
# @File     : Tools
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 工具脚本
# @History  :
#   2020/7/19 Dean First Release
import requests
import os

def sendMessage(data=None):
    """
    向企业我的微信发送信息
    """
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

if __name__ == '__main__':
    d = 'G:\\AI_dataset\\DOM\\Segmentation\\test' #获取当前文件夹下个数
    print(countNumOfFolder(d))