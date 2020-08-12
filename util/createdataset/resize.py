# Author:Dean
# Mail:dean0731@qq.com
# Time:2020-07-01
# Desc: 修改原始数据集
import os
from PIL import Image
src = r'E:\Workspace\PythonWorkSpace\dataset\dom\Segmentation'
des = r'E:\Workspace\PythonWorkSpace\dataset\dom\Segmentation5'
target_size=(288,288)
target_format = 'tif'
i = 0
for root, dirs, files in os.walk(src, topdown=False):
    temp = root.split('\\segmentation\\')
    for name in files:
        img = Image.open(os.path.join(root,name))
        img = img.resize(target_size,Image.ANTIALIAS)
        i=i+1
        name,format = name.split('.')
        newroot = os.path.join(des,temp[1])
        if 'tif' in format:
            format=target_format
        newname = name+'.'+format
        if not os.path.exists(newroot):
            os.makedirs(newroot)
        path = os.path.join(newroot,newname)
        img.save(path)
        print(i,path)
