# @Time     : 2020/8/26 17:23
# @File     : Config
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release
from utils.Path import DatasetPath
from pytorch.util.Dataset import Dataset
from torchvision import transforms
from pytorch.network.Segnet import Segnet
import numpy as np
import torch
input_size = (576,576)
in_channels = 3
out_channels = 2
target_size = input_size
batch_size = 2
path = DatasetPath('dom')
learning_rate = 1e-4
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Segnet(in_channels,out_channels)
loss = torch.nn.CrossEntropyLoss()
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

train_dataset = Dataset(path.getPath(),type='train',transform=transform,target_transform= target_transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
val_dataset = Dataset(path.getPath(),type='val',transform=transform,target_transform= target_transforms)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
test_dataset = Dataset(path.getPath(),type='test',transform=transform,target_transform= target_transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
