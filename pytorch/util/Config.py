# @Time     : 2020/8/26 17:23
# @File     : Config
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release
from pytorch.util.Dataset import Dataset
from torchvision import transforms
from pytorch.network import Segnet
import numpy as np
import torch
from util import flag
from util import get_dir
from util.cls import DatasetPath


def acc(y,y_pred):
    shape = y.shape
    return torch.true_divide(y_pred.argmax(dim=1).eq(y).sum(),(shape[1]*shape[2]))
def iou(y,y_pred):
    y_pred = y_pred.argmax(dim=1)
    iou = 0
    for y_,y_pred_ in zip(y,y_pred):
        a = y_.sum()
        b = y_pred_.sum()
        y_pred_[y_pred_==0]=2
        c = y_.eq(y_pred_).sum()
        iou = iou + torch.true_divide(c,(a+b-c))
    return iou

input_size = (576,576)
in_channels = 3
out_channels = 2
target_size = input_size
batch_size = 3
path = DatasetPath('dom')
learning_rate = 1e-4
num_epochs = int(flag.get('epochs') or 40)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Segnet.Segnet2(in_channels,out_channels)
loss = torch.nn.CrossEntropyLoss()
_,h5_dir,event_dir = get_dir("source/pytorch")
def toNumpy(x):
    return np.array(x,dtype=np.int)
def jiangwei(x):
    return torch.squeeze(x,dim=0)
transform = [
    transforms.Resize(input_size,interpolation=0),
    transforms.ToTensor()
]
target_transforms = [
    transforms.Resize(target_size,interpolation=0),
    transforms.Lambda(toNumpy),
    transforms.ToTensor(),
    transforms.Lambda(jiangwei)
]

train_dataset = Dataset(path.getPath(DatasetPath.TRAIN),transform=transform,target_transform= target_transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
val_dataset = Dataset(path.getPath(DatasetPath.VAL),transform=transform,target_transform= target_transforms)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
test_dataset = Dataset(path.getPath(DatasetPath.TEST),transform=transform,target_transform= target_transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
