from . import Dataset
import numpy as np
import os
from PIL import Image
class CountrySide(Dataset.Dataset):
    def __init__(self,parent,dir=('img', 'label_png'),shapeToOneDimension = False,data_size='tif_576'):
        Dataset.Dataset.__init__(self,parent,dir,shapeToOneDimension,data_size)

    def setDataset(self,flag='tif_3000'):
        if flag == 'tif_576':
            self.train_dir = r'dom/segmentation2/train'
            self.val_dir = r'dom/segmentation2/val'
            self.test_dir = r'dom/segmentation2/test'
            print('tif_576')
        elif flag == 'tif_288':
            self.train_dir = r'dom/segmentation5/train'
            self.val_dir = r'dom/segmentation5/val'
            self.test_dir = r'dom/segmentation5/test'
            print('tif_288')
        elif flag == 'png_576':
            self.train_dir = r'dom/segmentation3/train'
            self.val_dir = r'dom/segmentation3/val'
            self.test_dir = r'dom/segmentation3/test'
            print('png_576')
        elif flag == 'png_288':
            self.train_dir = r'dom/segmentation4/train'
            self.val_dir = r'dom/segmentation4/val'
            self.test_dir = r'dom/segmentation4/test'
            print('png_288')
        elif flag=='tif_3072':
            self.train_dir = r'dom/segmentation/train'
            self.val_dir = r'dom/segmentation/val'
            self.test_dir = r'dom/segmentation/test'
            print('tif_3072')
        else:
            print("未找到数据集")
            exit()
    def adjustData(self,img, mask, num_classes,flag):
        mask = mask[:, :, :, 0].astype(int) if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_classes,)).astype(int)  # (2, 416, 416, 2)
        for i, j in zip(range(num_classes), [0,38]):
            new_mask[:, :, :, i] = (j == mask[:, :, :])

        if flag:
            mask = new_mask.reshape((-1,new_mask.shape[1]*new_mask.shape[2],num_classes))
        else:
            mask = new_mask
        return (img, mask)
class CountrySide2(Dataset.Dataset):
    def __init__(self,parent,dir=('img', 'label_png'),shapeToOneDimension = False,data_size='tif_576'):
        Dataset.Dataset.__init__(self,parent,dir,shapeToOneDimension,data_size)

    def getData(self,target_size=(64, 64),mask_size=(64,64),batch_size = 4):
        with open(os.path.join(self.parent,'dom/data.txt'),encoding='utf-8') as f:
            lines = f.readlines()
        length = len(lines)
        train,val,test= lines[0:int(length/10*8)],lines[int(length/10*8):int(length/10*9)],lines[int(length/10*9):length]

        train_img = self.getGernerator(type='img',size=target_size,batch_size=batch_size,lines=train)
        train_label = self.getGernerator(type='label_png',size=target_size,batch_size=batch_size,lines=train)
        val_img = self.getGernerator(type='img',size=target_size,batch_size=batch_size,lines=val)
        val_label = self.getGernerator(type='label_png',size=target_size,batch_size=batch_size,lines=val)
        test_img = self.getGernerator(type='img',size=target_size,batch_size=batch_size,lines=test)
        test_label = self.getGernerator(type='label_png',size=target_size,batch_size=batch_size,lines=test)

        return (train_img,train_label),(val_img,val_label),(test_img,test_label)

    def getGernerator(self,type='img',size=(64,64),batch_size=4,num_classes=2,lines=None):
        temp = lines
        lines = []
        if type=='img':
            for k in temp:
                lines.append(os.path.join(self.parent,k.strip().split(';')[0]))
            i=0
            while 1:
                img =[]
                img2 = []
                for _ in range(batch_size):
                    image = Image.open(lines[i])

                    images = image.resize(size)
                    images = np.array(images)
                    img.append(images)

                    images = images.resize((3072,3072))
                    images = np.array(images)
                    img2.append(images)

                    if i == (len(lines)-1):
                        i=0
                    else:
                        i=i+1
                yield [np.array(img),np.array(img2)]
        elif type=='label_png':
            for k in temp:
                lines.append(os.path.join(self.parent,k.strip().split(';')[1]))
            i=0
            while 1:
                img =[]
                for _ in range(batch_size):
                    images = Image.open(lines[i])
                    images = images.resize(size)
                    images = np.array(images)
                    new_images = np.zeros(images.shape + (num_classes,)).astype(int)  # (w,h,num_classes)
                    for m, n in zip(range(num_classes), [0,1]):
                        new_images[:, :, m] = (n == images[:, :])
                    img.append(new_images)
                    if i == (len(lines)-1):
                        i=0
                    else:
                        i=i+1
                yield np.array(img)
        else:
            print("根据文件创作生成器失败")
            exit()
if __name__ == '__main__':
    dataset = CountrySide(parent= './dataset',data_size='tif_576')
    print(dataset.train_dir)
    data,validation_data,test_data = dataset.getData(target_size=(576,576),mask_size=(576,576),batch_size=4)
    data.__next__()