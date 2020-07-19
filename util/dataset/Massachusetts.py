from . import Dataset


class Massachusetts(Dataset.Dataset):
    def __init__(self,parent,dir=('Input images', 'Target maps'),shapeToOneDimension = False,data_size='tif_576'):
        Dataset.Dataset.__init__(self,parent,dir,shapeToOneDimension,data_size)

    def setDataset(self,flag='tif_576'):
        if flag== 'tif_576':
            self.train_dir = r'Massachusetts576/Training Set'
            self.val_dir = r'Massachusetts576/Validation Set'
            self.test_dir = r'Massachusetts576/Test Set'
            print('tif_576')
        elif flag=='tif_1500':
            self.train_dir = r'Massachusetts1500/Training Set'
            self.val_dir = r'Massachusetts1500/Validation Set'
            self.test_dir = r'Massachusetts1500/Test Set'
            print('tif_1500')
        else:
            print("未找到数据集")
            exit()