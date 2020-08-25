import os
a = os.listdir(r"G:\AI_dataset\DOM\大郭楼-DOM\image-3000\json")
with open("train2.txt", "w", encoding="utf8") as f:
    for i in a:
        if(i.endswith("png")):
            f.write(i.split(".")[0]+".tif"+";"+i+"\n")

