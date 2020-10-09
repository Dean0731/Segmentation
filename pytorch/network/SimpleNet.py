# @Time     : 2020/8/26 12:34
# @File     : SimpleNet
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 自定义简单2层网络，用于测试
# @History  :
#   2020/8/26 Dean First Release
import torch.nn as nn
import torch.nn.functional as F

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
if __name__ == '__main__':
    print(callable(Net()))