# @Time     : 2020/8/26 17:23
# @File     : Config
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release
from pytorch.util.Dataset import Dataset
from pytorch.util import Transform
import torch.optim as optim
from pytorch.network import Segnet,Deeplabv3
import torch
import os

import util
from util import flag
from util import get_dir
from util.cls import DatasetPath

def acc(y_pred,y):
    shape = y.shape
    return torch.true_divide(y_pred.argmax(dim=1).eq(y).sum(),(shape[1]*shape[2]))
def iou(y_pred,y):
    y_pred = y_pred.argmax(dim=1)
    iou = 0
    for y_,y_pred_ in zip(y,y_pred):
        a = y_.sum()
        b = y_pred_.sum()
        y_pred_[y_pred_==0]=2
        c = y_.eq(y_pred_).sum()
        iou = iou + torch.true_divide(c,(a+b-c))
    return iou

metrics = [acc,iou]
input_size = (576,576)
in_channels = 3
out_channels = 2
target_size = input_size
batch_size = 3
path = DatasetPath('dom')
learning_rate = 1e-4
num_epochs = int(flag.get('epochs') or 80)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Segnet.Segnet2(in_channels,out_channels).to(device)
model = Deeplabv3.deeplabv3_resnet50(num_classes=2).to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()
log = bool(flag.get('log') or True)

if log:
    log_,h5_dir,event_dir = get_dir(os.path.join(util.getParentDir(),'source/pytorch'))
    with open(os.path.join(log_,"config.txt"),"w")as f:
        f.write("metrice:{}\n".format(metrics))
        f.write("input_size:{}\n".format(input_size))
        f.write("target_size:{}\n".format(target_size))
        f.write("in_channels:{}\n".format(in_channels))
        f.write("out_channels:{}\n".format(out_channels))
        f.write("batch_size:{}\n".format(batch_size))
        f.write("datasetPath:{}\n".format(path))
        f.write("learningRate:{}\n".format(learning_rate))
        f.write("num_epochs:{}\n".format(num_epochs))
        f.write("device:{}\n".format(device))
        f.write("model:{}\n".format(model._get_name()))
        f.write("optimizer:{}\n".format(optimizer))
        f.write("loss:{}\n".format(loss))
else:
    h5_dir,event_dir = False,False


train_dataset = Dataset(path.getPath(DatasetPath.TRAIN),transform=Transform.getTransforms(input_size),target_transform= Transform.getTargetTransforms(target_size))
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
val_dataset = Dataset(path.getPath(DatasetPath.VAL),transform=Transform.getTransforms(input_size),target_transform= Transform.getTargetTransforms(target_size))
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
test_dataset = Dataset(path.getPath(DatasetPath.TEST),transform=Transform.getTransforms(input_size),target_transform= Transform.getTargetTransforms(target_size))
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

