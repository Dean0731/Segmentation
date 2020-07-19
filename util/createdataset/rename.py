# data:2020-01-06
# user:dean
# desc:文件重命名
import os
dom = ""
target_dir = r"I:\AI_dataset\DOM\郭土楼44-dom\image-3000"
files_name = os.listdir(target_dir)
prefix = "郭土楼44-dom"
for name,i in zip(files_name,range(len(files_name))):
    new_name = "{}_{}.tif".format(prefix,str(i).rjust(5,'0'))
    # print(name,new_name)
    os.renames(os.path.join(target_dir,name),os.path.join(target_dir,new_name))
