import os
from PIL import Image
from random import randint
def rgbTo256(src,filter):
    """
    将文件夹中的rgb图片，转换为256色图片，labelme的标签格式
    只能转换png格式，
    @params: src
    """
    def randomPalette(length, min, max):
        return [randint(min, max) for x in range(length)]
    k = 1
    for root, dirs, files in os.walk(src, topdown=False):
        for name in files:
            path = os.path.join(root,name)
            if filter in path:
                img = Image.open(path)
                img = img.convert('P')
                i = randomPalette(0, 0, 0)
                img.putpalette(i)
                img.save(path)
                print(k,path)
                k = k+1
def reformate(src):
    """
    将源文件改编为png格式
    """
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