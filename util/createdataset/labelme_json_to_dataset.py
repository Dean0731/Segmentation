# data:2020-01-04
# user:dean
# desc:批量将json文件转为 label
import os
# linux_dir = r"/media/dean/Document/AI_dataset/DOM/裴庄村51-dom/image-3000"
windows_dir = r"I:\AI_dataset\DOM\米庄村-DOM\image-3000\json"
dir = windows_dir
files = [os.path.join(dir,file) for file in os.listdir(dir) if file.endswith(".json")]
for file in files:
    cmd = "labelme_json_to_dataset {}".format(file)
    print(cmd)
    os.system(cmd)