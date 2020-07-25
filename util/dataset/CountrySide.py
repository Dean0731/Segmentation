from . import Dataset
import numpy as np

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