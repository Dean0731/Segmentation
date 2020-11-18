import requests,logging
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