# @Time     : 2020/8/25 18:36
# @File     : Config
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release

from util.Path import DatasetPath
import torch
from torchvision import datasets,transforms
from pytorch.network import SimpleNet

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=4,
    pin_memory=True, # 可以加速计算
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=1,
    pin_memory=True, # 可以加速计算
)
model = SimpleNet.Net()
