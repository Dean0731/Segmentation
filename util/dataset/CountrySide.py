from . import Dataset


class CountrySide(Dataset.Dataset):
    def __init__(self,parent,dir=('img', 'label_png'),shapeToOneDimension = False,data_size='tif_576'):
        Dataset.Dataset.__init__(self,parent,dir,shapeToOneDimension,data_size)

    def setDataset(self,flag='tif_3000'):
        if flag == 'tif_576':
            self.train_dir = r'segmentation2/train'
            self.val_dir = r'segmentation2/val'
            self.test_dir = r'segmentation2/testNetwork'
            print('tif_576')
        elif flag == 'tif_288':
            self.train_dir = r'segmentation5/train'
            self.val_dir = r'segmentation5/val'
            self.test_dir = r'segmentation5/testNetwork'
            print('tif_288')
        elif flag == 'png_576':
            self.train_dir = r'segmentation3/train'
            self.val_dir = r'segmentation3/val'
            self.test_dir = r'segmentation3/testNetwork'
            print('png_576')
        elif flag == 'png_288':
            self.train_dir = r'segmentation4/train'
            self.val_dir = r'segmentation4/val'
            self.test_dir = r'segmentation4/testNetwork'
            print('png_288')
        elif flag=='tif_3072':
            self.train_dir = r'Segmentation/train'
            self.val_dir = r'Segmentation/val'
            self.test_dir = r'Segmentation/testNetwork'
            print('tif_3072')
        else:
            print("未找到数据集")
            exit()