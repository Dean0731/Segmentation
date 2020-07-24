from util.dataset import Dataset
import numpy as np
class AerialImage(Dataset.Dataset):
    def __init__(self,parent,dir=('images', 'gt'),shapeToOneDimension = False,data_size='tif_576'):
        Dataset.Dataset.__init__(self,parent,dir,shapeToOneDimension,data_size)

    def setDataset(self,flag='tif_576'):
        if flag == 'tif_576':
            self.train_dir = r'aerialImage/AerialImage576/train'
            self.val_dir = r'aerialImage/AerialImage576/val'
            self.test_dir = r'aerialImage/AerialImage576/test'
            print('tif_576')
        elif flag =='tif_3072':
            self.train_dir = r'aerialImage/AerialImage3072/train'
            self.val_dir = r'aerialImage/AerialImage3072/val'
            self.test_dir = r'aerialImage/AerialImage3072/test'
            print('tif_3072')
        else:
            print("未找到数据集")
            exit()
    def adjustData(self,img, mask, num_classes,flag):
        mask = mask[:, :, :, 0].astype(int) if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_classes,)).astype(int)  # (2, 416, 416, 2)
        for i, j in zip(range(num_classes), [0,255]):
            new_mask[:, :, :, i] = (j == mask[:, :, :])
        if flag:
            mask = new_mask.reshape((-1,new_mask.shape[1]*new_mask.shape[2],num_classes))
        else:
            mask = new_mask
        return (img, mask)
if __name__ == '__main__':
    pass