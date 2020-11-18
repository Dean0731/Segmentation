import os,sys,datetime
def printAndWriteAttr(obj,path):
    if os.path.exists(path):
        with open(os.path.join(path,"config.txt"),"w")as f:
            for attr in dir(obj):
                if not attr.startswith("__"):
                    f.write("{}:{}\n".format(attr,getattr(obj,attr)))
    else:
        raise FileExistsError("{}:不存在".format(path))

def getParentDir():
    return os.path.dirname(sys.path[0])


def countNumOfFolder(path):
    file = 0
    folder = 0
    for root, dirs, files in os.walk(path, topdown=False):
        file = file + len(files)
        folder = folder + len(dirs)
    return [file,folder]


def get_dir(parent=None):
    """ 生成目录
    工具脚本，生成文件目录，
    ~/file-data-time
        -event
        -h5
    """
    time = datetime.datetime.now()
    if parent == None:
        parent = 'source'
    log_dir = os.path.join(parent,'logs-{}-{}-{}-{}-{}-{}'.format(time.year,time.month,time.day,time.hour,time.minute,time.second))
    os.makedirs(log_dir)
    return log_dir