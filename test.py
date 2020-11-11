import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = ["/home/yqw/k-means/voc", "/home/yqw/k-means/zz",
                     "/home/yqw/k-means/ua-detrac"
                    ]
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
temp = [[] for i in range(len(ANNOTATIONS_PATH))]
for i,path in enumerate(ANNOTATIONS_PATH):
    data = load_dataset(path)
    for cluster in range(1,13):
        out = kmeans(data, k=cluster)
        temp[i].append(avg_iou(data, out))
        print(path,cluster)
print(temp)
