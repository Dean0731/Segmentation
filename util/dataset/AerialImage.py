from util.dataset import Dataset


class AerialImage(Dataset.Dataset):
    def __init__(self,parent,dir=('images', 'gt'),shapeToOneDimension = False,data_size='tif_576'):
        Dataset.Dataset.__init__(self,parent,dir,shapeToOneDimension,data_size)
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

if __name__ == '__main__':
    pass