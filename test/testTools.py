import os
import numpy as np
from PIL import Image
def compareFolder(file,file2):
    image = file
    label = file2
    image = os.listdir(image)
    label = os.listdir(label)
    print(len(label))
    print(image==label)

def compareFolder(file_less,file_much):
    """
    比较两个文件夹的差积 file_much-file_less
    """
    image = os.listdir(file_less)
    label = os.listdir(file_much)
    print(len(image))
    print(len(label))
    i,j=0,0
    with open('temp.txt','w') as f:
        for _ in label:
            if label[i]==image[j]:
                i = i+1
                j = j+1
            else:
                f.write(label[i]+"\n")
                i = i+1
def getNumOfImage(file):
    img = Image.open(file)
    print(img.mode)
    img = img.convert('L')
    # img = img.resize((576,500))
    print(img.mode)
    img = np.asarray(img)
    print(img.shape)

    for i in range(576):
        for j in range(576):
            print(img[i,j],end='')
        print()
if __name__ == '__main__':
    file = r'G:\AI_dataset\aerialImage\AerialImage576\test\gt\austin15_006.tif'
    # file = r'G:\AI_dataset\massachusetts\Massachusetts576\Test Set\Target maps\22828930_15_000.tif'
    # file = r'G:\AI_dataset\massachusetts\Massachusetts576\Test Set\Target maps\22828930_15_000.tif'
    # file = r'G:\AI_dataset\dom\Segmentation\val\label_png\裴庄村51-dom\裴庄村51-dom_00001.png'
    getNumOfImage(file)