# coding:utf-8
# file: rename2.py
# author: Dean
# contact: 1028968939@qq.com
# time: 2020/1/18 16:12
# desc:
import os
dir = r"I:\AI_dataset\DOM\勤新村-5-DOM\image-3000"
files = os.listdir(dir)

for file in files:
    sub = file.split(".")[1]
    temp = file.split(".")[0]
    name = temp.split("_")[0]
    i = temp.split("_")[1]
    new_num = i.rjust(5,'0')

    src = os.path.join(dir,file)
    new_name = os.path.join(dir,"{}_{}.{}".format(name,new_num,sub))
    os.rename(src,new_name)
    # print(src,new_name)