# import numpy,os
# dir = 'E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation'
# with open(os.path.join(dir,'data.txt'),'r',encoding='utf-8') as f:
#     lines = f.readlines()
#     numpy.random.seed(7)
#     numpy.random.shuffle(lines)
# temp = int(len(lines)*0.02)
# temp2 = int(len(lines)*0.04)
# with open(os.path.join(dir,'data_train.txt'),'w',encoding='utf-8') as f1:
#     f1.writelines(lines[0:temp])
# with open(os.path.join(dir,'data_val.txt'),'w',encoding='utf-8') as f2:
#     f2.writelines(lines[temp:temp2])
# with open(os.path.join(dir,'data_test.txt'),'w',encoding='utf-8') as f3:
#     f3.writelines(lines[temp2:])
# with open(os.path.join(dir,'data_label.txt'),'w',encoding='utf-8') as f4:
#     f4.write('bg\n')
#     f4.write('house')
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
target_size = 1024
data_dir='/home/aistudio/data/data51222/'
train_list='E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation\data_train.txt'
val_list='E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation\data_val.txt'
test_list='E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation\data_test.txt'
label_list='E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation\data_label.txt'
from paddlex.seg import transforms
transforms = transforms.Compose([
    transforms.Resize(target_size=target_size),
    transforms.Normalize()
])

# 定义数据集
# train_dataset = pdx.datasets.SegDataset(
#     data_dir=data_dir,
#     file_list=train_list,
#     label_list=label_list,
#     transforms=transforms,
#     shuffle=True)
val_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list=val_list,
    label_list=label_list,
    transforms=transforms)
# test_dataset = pdx.datasets.SegDataset(
#     data_dir=data_dir,
#     file_list=test_list,
#     label_list=label_list,
#     transforms=transforms)
for i in val_dataset:
    print(i)