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
            name = str(obj.findtext("name"))
            xmin = int(obj.findtext("bndbox/xmin"))
            ymin = int(obj.findtext("bndbox/ymin"))
            xmax = int(obj.findtext("bndbox/xmax"))
            ymax = int(obj.findtext("bndbox/ymax"))
            dataset.append([name,xmin,ymin,xmax,ymax])
    return dataset
path = r'G:'
data = load_dataset(path)
img = cv2.imread(os.path.join(path,'1.jpg'))
for i in data:
    name = i[0]
    x = i[1]
    y = i[2]
    x_h = i[3]
    y_h = i[4]
    if name == 'car':
        color = (0,255,0)
    elif name =='bus':
        color = (0,0,255)
    else:
        color = (0,0,0)
    cv2.rectangle(img, (x,y),(x_h,y_h),color,2)
    cv2.rectangle(img, (x,y-6),(x+20,y), color,-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, name, (x,y), font, 0.3, (0,0,0), 0)
cv2.imwrite(r"G:\1-1.png",img)