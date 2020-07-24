from . import Dataset
import numpy as np
class Massachusetts(Dataset.Dataset):
    def __init__(self,parent,dir=('Input images', 'Target maps'),shapeToOneDimension = False,data_size='tif_576'):
        Dataset.Dataset.__init__(self,parent,dir,shapeToOneDimension,data_size)

    def setDataset(self,flag='tif_576'):
        if flag== 'tif_576':
            self.train_dir = r'massachusetts/Massachusetts576/Training Set'
            self.val_dir = r'massachusetts/Massachusetts576/Validation Set'
            self.test_dir = r'massachusetts/Massachusetts576/Test Set'
            print('tif_576')
        elif flag=='tif_1500':
            self.train_dir = r'massachusetts/Massachusetts1500/Training Set'
            self.val_dir = r'massachusetts/Massachusetts1500/Validation Set'
            self.test_dir = r'massachusetts/Massachusetts1500/Test Set'
            print('tif_1500')
        else:
            print("未找到数据集")
            exit()
    def adjustData(self,img, mask, num_classes,flag):
        mask = mask[:, :, :, 0].astype(int) if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_classes,)).astype(int)  # (2, 416, 416, 2)
        for i, j in zip(range(num_classes), [0,76]):
            new_mask[:, :, :, i] = (j == mask[:, :, :])
        if flag:
            mask = new_mask.reshape((-1,new_mask.shape[1]*new_mask.shape[2],num_classes))
        else:
            mask = new_mask
        return (img, mask)