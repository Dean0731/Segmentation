# Author:Dean
# Mail:dean0731@qq.com
# Time:2020-07-01
# Desc: 修改数据的大小或格式
import os
from PIL import Image

def main(src,des,target_size,target_format):
    i = 0
    for root, dirs, files in os.walk(src, topdown=False):
        temp = root.split('\\segmentation\\')
        for name in files:
            img = Image.open(os.path.join(root,name))
            if target_size != None:
                img = img.resize(target_size,resample=Image.NEAREST)
            i=i+1
            name,format = name.split('.')
            newroot = os.path.join(des,temp[1])
            if target_format !=None:
                if 'tif' in format:
                    format=target_format
            newname = name+'.'+format
            if not os.path.exists(newroot):
                os.makedirs(newroot)
            path = os.path.join(newroot,newname)
            # img.save(path)
            print(i,path)

def reformate(src):
    i = 0
    for root, dirs, files in os.walk(src, topdown=False):
        for name in files:
            img = Image.open(os.path.join(root,name))
            i=i+1
            name,format = name.split('.')
            newname = name+'.png'
            path = os.path.join(root,newname)
            img.save(path)
            print(i,path)

if __name__ == '__main__':
    src = r'G:\AI_dataset\dom\segmentation'
    des = r'G:\AI_dataset\dom\segmentation3'
    target_format = 'png'
    # main(src,des,target_size=None,target_format=target_format)
    reformate("D:\desktop\img")

