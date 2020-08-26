# @Time     : 2020/8/26 17:20
# @File     : train
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release

from pytorch.util import Config

import torch
import torch.optim as optim
from pytorch.util import Config

from utils import Tools
print("Pytorch Version",torch.__version__)
def train(model,device,train_dataloader,optimizer,epoch):
    model.train()
    for idx,(data,target) in enumerate(train_dataloader):
        print("开始迭代")
        data,target = data.to(device),target.to(device)
        pred = model(data) # batch_size * 10
        loss = Config.loss(pred,target.long()) # 交叉熵损失函数
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Train Epoch:{}, iteration:{}, loss:{}".format(epoch,idx,loss))
def test(model,device,test_dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx,(data,target) in enumerate(test_dataloader):
            data,target = data.to(device),target.to(device)
            pred = model(data) # batch_size * 10
            total_loss += Config.loss(pred,target,reduction="sum").item()
        print("Test loss:{}".format(total_loss/len(test_dataloader.dataset)))

@Tools.Decorator.sendMessage()
@Tools.Decorator.timer()
def main():
    model = Config.model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    for epoch in range(Config.num_epochs):
        train(model, Config.device, Config.train_dataloader, optimizer, epoch)
        test(model, Config.device, Config.test_dataloader)

# torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    main()
