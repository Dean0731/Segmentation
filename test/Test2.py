import os
ft = r'G:\AI_dataset\马萨诸塞州-房屋数据集1\Massachusetts Buildings Dataset2\Training Set\Input images'
with open("temp.txt",'r')as f:
    for i in f.readlines():
        path = os.path.join(ft,i.strip())
        os.remove(path)