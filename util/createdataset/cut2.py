# Pragram:
#     数据集切割脚本，法国遥感房屋数据集 https://project.inria.fr/aerialimagelabeling/contest/
# History:
# 2020-07-17 Dean First Release
# Email:dean0731@qq.com
import tifffile as tiff  # 也可使用pillow或opencv 但若图片过大时可能会出问题
import os
width = 576   # 切割图像大小
height = 576  # 切割图像大小
iou =  228  # 交叉切割
file_home = r"G:\AI_dataset\马萨诸塞州-房屋数据集1\Massachusetts Buildings Dataset\Training Set\Target maps"
file_names = os.listdir(file_home)
target_home = r'G:\AI_dataset\马萨诸塞州-房屋数据集1\Massachusetts Buildings Dataset2\Training Set\Target maps'
if not os.path.exists(target_home):
    os.makedirs(target_home)

for i,file in enumerate(file_names):
    img = tiff.imread(os.path.join(file_home,file))  # 导入图片
    print("第{}导入图片完成".format(i),img.shape) # 原始图片大小
    pic_width = img.shape[1]
    pic_height = img.shape[0]
    col,col_end,row,row_end=0,0,0,0
    k = 0
    while True:
        row=row_end
        row_end = row_end+height
        if row_end > pic_height:
            break
        while True:
            col=col_end
            col_end=col_end+width
            if col_end > pic_width:
                break;
            cropped=img[col:col_end,row:row_end]
            name = "{}_{}.tif".format(file.split(".")[0],str(k).rjust(3,'0'))
            image_path = os.path.join(target_home,name)
            tiff.imsave(image_path, cropped)
            print("第{}图片的第{}个分割:{}".format(i,k,image_path))
            col_end=col_end-iou
            k = k+1
        row_end = row_end-iou
        col,col_end=0,0
