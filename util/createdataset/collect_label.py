# data:2020-01-04
# user:dean
# desc:将label文件夹中的laebl提取出来
import tifffile as tiff
from PIL import Image
import os
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
                break;