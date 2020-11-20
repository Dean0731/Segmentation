# @Time     : 2020/8/25 17:44
# @File     : pytorch_for_cnn
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch.example import Config

from util import func
print("Pytorch Version",torch.__version__)
def train(model,device,train_dataloader,optimizer,epoch):
    model.train()
    for idx,(data,target) in enumerate(train_dataloader):
        data,target = data.to(device),target.to(device)
        pred = model(data) # batch_size * 10
        loss = F.nll_loss(pred,target) # 交叉熵损失函数
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if idx % 100 ==0:
            print("Train Epoch:{}, iteration:{}, loss:{}".format(epoch,idx,loss))
def test(model,device,test_dataloader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for idx,(data,target) in enumerate(test_dataloader):
            data,target = data.to(device),target.to(device)
            pred = model(data) # batch_size * 10
            total_loss += F.nll_loss(pred,target,reduction="sum").item()
            correct += pred.argmax(dim=1).eq(target).sum().item()
        print("temp Accuracy:{}, loss:{}".format(correct/len(test_dataloader.dataset),total_loss/len(test_dataloader.dataset)))
@func.Decorator.sendMessage()
@func.Decorator.timer()
def main():
    model = Config.model.to(Config.device)
    optimizer =optim.SGD(model.parameters(), lr=Config.learning_rate, momentum=Config.momentum)
    for epoch in range(Config.num_epochs):
        train(model, Config.device, Config.train_dataloader, optimizer, epoch)
    test(model, Config.device, Config.test_dataloader)

# torch.save(model.state_dict(),"last.pt")


if __name__ == '__main__':
    main()