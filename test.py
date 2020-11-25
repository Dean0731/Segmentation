import csv
import os
import sys
import numpy as np
import copy
import shutil
import pandas as pd
from collections import Counter
from shutil import copyfile
import cv2

path = os.getcwd()
print(path)
path_1 = path + '/' + 'data_error_0813'
list_name = os.listdir(path_1)
for n in list_name:
    if n[-3:] == 'csv':
        csvpath = path_1 + '/' + n
        imgpath = path_1 + '/' + n[:-3] + 'JPG'
        print(imgpath)
        if not os.path.exists(imgpath):
            print("nothing")

        filehand = open(csvpath,'r')
        csvlist = filehand.readlines()
        mark = []
        image = []
        count = 1


        for m in csvlist[1:]:
            m_split = m.split(',')
            xy = [m_split[2], m_split[3]]
            mark.append(xy)
            image = cv2.imread(imgpath)
            print("type:",type(image))
            first_point = (int(m_split[2])-50,int(m_split[3])-50)
            last_point = (int(m_split[2])+50,int(m_split[3])+50)
            cv2.rectangle(image, first_point, last_point, (0,255,0),2)
            cv2.imwrite(imgpath,image)
            print("标记次数",count)
            count = count + 1

    else:
        continue
    print(mark)



import glob
import xml.etree.ElementTree as ET

def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        try:
            tree = ET.parse(xml_file)
        except Exception as e:
            print(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height
            if (xmax - xmin)>0 and (ymax - ymin) >0:
                dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)
