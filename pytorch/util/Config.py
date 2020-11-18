# @Time     : 2020/8/26 17:23
# @File     : Config
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release
import torch.optim as optim
from pytorch.util.Dataset import Dataset
from pytorch.util import Transform
from pytorch.util.Metrice import Segmentation
from pytorch.network import Segnet,Deeplabv3
import torch
import os,sys
from util import flag
from util import folder
from util import DatasetPath

metrics = [Segmentation.getAcc,Segmentation.getIou,Segmentation.getRecall,Segmentation.getPrecision]
input_size = (512,512)
in_channels = 3
out_channels = 2
target_size = input_size
batch_size = 4
path = DatasetPath('dom')
learning_rate = 1e-4
num_epochs = int(flag.get('epochs') or 80)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Segnet.Segnet2(in_channels,out_channels).to(device)
model = Deeplabv3.deeplabv3_resnet50(num_classes=2).to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()


if os.path.exists(str(flag.get('log'))):
    log = flag.get('log')
    folder.printAndWriteAttr(sys.modules[__name__],log)
else:
    log = False


train_dataset = Dataset(path.getPath(DatasetPath.TRAIN),transform=Transform.getTransforms(input_size),target_transform= Transform.getTargetTransforms(target_size))
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
val_dataset = Dataset(path.getPath(DatasetPath.VAL),transform=Transform.getTransforms(input_size),target_transform= Transform.getTargetTransforms(target_size))
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
test_dataset = Dataset(path.getPath(DatasetPath.TEST),transform=Transform.getTransforms(input_size),target_transform= Transform.getTargetTransforms(target_size))
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

train_dataloader = test_dataloader

