# @Time     : 2020/7/19 14:19
# @File     : GenerateDir
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 工具脚本
# @History  :
#   2020/7/19 Dean First Release


import os
import datetime


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