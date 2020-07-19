# desc: 将png 标签转化为单通道 黑白标签 并转化为tif

import os
from PIL import Image
threshold = 0
table = []
for i in range(256):
    if i > threshold:
        table.append(255)
    else:
        table.append(0)
target_dir = r"I:\AI_dataset\DOM\米庄村-DOM\image-3000\json"
files = [os.path.join(target_dir,file) for file in os.listdir(target_dir) if file.endswith(".png")]
for file in files:
    image_file_name = os.path.basename(file)
    num = image_file_name.split(".")[0]
    image_file = Image.open(file)  # open colour image
    # image_file = image_file.convert('L') # convert image to black and white
    image_file = image_file.point(table, '1')
    new_file = os.path.join(target_dir,"{}.tif".format(num))
    image_file.save(new_file)
    print(new_file)

