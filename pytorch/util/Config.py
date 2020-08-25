# @Time     : 2020/8/25 18:36
# @File     : Config
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
from util.Path import DatasetPath
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(4*4*50,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,4*4*50)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

train_dataset = datasets.MNIST(
    DatasetPath('mnist').getPath(),
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307),(0.3081))
    ]))
test_dataset = datasets.MNIST(
    DatasetPath('mnist').getPath(),
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307),(0.3081))
    ]))
batch_size = 32
learning_rate = 1e-2
momentum = 0.5
num_epochs = 2

