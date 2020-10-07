# @Time     : 2020/8/30 22:43
# @File     : generate_data_txt
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/30 Dean First Release

import os
import numpy as np
from PIL import Image
import tifffile as tiff  # 也可使用pillow或opencv 但若图片过大时可能会出问题
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
def generateDataTxT(src,img,label):
    with open(os.path.join(src,'data.txt'),'w',encoding='utf-8') as f:
        for root, dirs, files in os.walk(src, topdown=False):
            for name in files:
                if not label in root and not 'data.txt' in name:
                    root = root.replace(src,'')
                    f.write("{};{}\n".format(os.path.join(root,name),os.path.join(root.replace(img,label),name)))

#-------------------------------------------------------------------
# 数据集切割脚本，法国遥感房屋数据集 https://project.inria.fr/aerialimagelabeling/contest/

def cut():
    width = 576   # 切割图像大小
    height = 576  # 切割图像大小
    iou =  228  # 交叉切割
    file_home = r"G:\AI_dataset\马萨诸塞州-房屋数据集1\Massachusetts Buildings Dataset\Training Set\Target maps"
    file_names = os.listdir(file_home)
    target_home = r'G:\AI_dataset\马萨诸塞州-房屋数据集1\Massachusetts Buildings Dataset2\Training Set\Target maps'
    if not os.path.exists(target_home):
        os.makedirs(target_home)

    for i,file in enumerate(file_names):
        img = tiff.imread(os.path.join(file_home,file))  # 导入图片
        print("第{}导入图片完成".format(i),img.shape) # 原始图片大小
        pic_width = img.shape[1]
        pic_height = img.shape[0]
        col,col_end,row,row_end=0,0,0,0
        k = 0
        while True:
            row=row_end
            row_end = row_end+height
            if row_end > pic_height:
                break
            while True:
                col=col_end
                col_end=col_end+width
                if col_end > pic_width:
                    break;
                cropped=img[col:col_end,row:row_end]
                name = "{}_{}.tif".format(file.split(".")[0],str(k).rjust(3,'0'))
                image_path = os.path.join(target_home,name)
                tiff.imsave(image_path, cropped)
                print("第{}图片的第{}个分割:{}".format(i,k,image_path))
                col_end=col_end-iou
                k = k+1
            row_end = row_end-iou
            col,col_end=0,0
#-----------------------------------------

# desc:将label文件夹中的laebl提取出来
def get():
    target_dir = r"I:\AI_dataset\DOM\米庄村-DOM\image-3000\json"  # json_label 所在的文件夹
    files = [os.path.join(target_dir,file) for file in os.listdir(target_dir)]
    for i in files:
        if os.path.isdir(i):
            lables = os.listdir(i)
            for file in lables:
                if file == "label.png":
                    image_path = os.path.join(i, "label.png")
                    imgae = Image.open(image_path)
                    parent_dir_name = os.path.basename(os.path.dirname(image_path))
                    new_name = "{}.png".format(parent_dir_name.rsplit("_",1)[0])
                    imgae.save(os.path.join(target_dir,new_name))
                    print("第{}个文件夹".format(i))
                    break
# --------------------------------------
# desc:批量将json文件转为 label
# linux_dir = r"/media/dean/Document/AI_dataset/DOM/裴庄村51-dom/image-3000"
def change():
    windows_dir = r"I:\AI_dataset\DOM\米庄村-DOM\image-3000\json"
    dir = windows_dir
    files = [os.path.join(dir,file) for file in os.listdir(dir) if file.endswith(".json")]
    for file in files:
        cmd = "labelme_json_to_dataset {}".format(file)
        print(cmd)
        os.system(cmd)


# ------------------------------------------------------------
# Desc: 修改数据的大小或格式
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
def generateTrainValTest(dir):
    with open(os.path.join(dir,'data.txt'),'r',encoding='utf-8') as f:
        lines = f.readlines()
    np.random.seed(7)
    np.random.shuffle(lines)
    temp = int(len(lines)*0.8)
    temp2 = int(len(lines)*0.9)
    with open(os.path.join(dir,'data_train.txt'),'w',encoding='utf-8') as f1:
        f1.writelines(lines[0:temp])
    with open(os.path.join(dir,'data_val.txt'),'w',encoding='utf-8') as f2:
        f2.writelines(lines[temp:temp2])
    with open(os.path.join(dir,'data_test.txt'),'w',encoding='utf-8') as f3:
        f3.writelines(lines[temp2:])
    with open(os.path.join(dir,'data_label.txt'),'w',encoding='utf-8') as f4:
        f4.write('bg\n')
        f4.write('house')
if __name__ == '__main__':

    data_txt_dir = r'C:\Users\root\Desktop'
    # generateDataTxT(data_txt_dir,'img','label_png')
    generateTrainValTest(data_txt_dir)