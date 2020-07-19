# data:2020-01-04
# user:dean
# desc:图像切割脚本
import tifffile as tiff  # 也可使用pillow或opencv 但若图片过大时可能会出问题
import os
width = 576   # 切割图像大小
height = 576  # 切割图像大小
jiaocha =  int(width*0.1)   # 交叉切割
home = "/media/dean/Document/AI_dataset/DOM/"
file_name = "裴庄村51-dom"
image_dir = os.path.join(home,file_name)
image = os.path.join(image_dir,file_name+".tif")
target_dir = os.path.join(image_dir,"image-"+str(width))  # 切割后图片存储位置
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
img = tiff.imread(image)  # 导入图片
print("导入图片完成",img.shape) # 原始图片大小
pic_width = img.shape[1]
pic_height = img.shape[0]
row_num = pic_width//width  # 纵向切割数量
col_num = pic_height // height  # 横向切割数量
print("开始进行切割，可切割总数为{}".format(col_num*row_num))
for j in range(col_num):
    for i in range(row_num):
        num = j * row_num + i
        print("正在进行第{}张切割".format(num + 1))
        row = i * width
        row_end = row + width
        col = j * height
        col_end = col + height
        # print(col,col_end,row,row_end)
        cropped = img[col:col_end,row:row_end]
        name = "{}_{}.tif".format(file_name,num)
        image_path = os.path.join(target_dir,name)
        tiff.imsave(image_path, cropped)