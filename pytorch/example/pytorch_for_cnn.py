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
from pytorch.util import Config
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
            correct += pred.argmax(dim=1).eq(target.view_as(pred)).sum().item()
        print("test Accuracy:{}, loss:{}".format(correct/len(test_dataloader.dataset),total_loss/len(test_dataloader.dataset)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataloader = torch.utils.data.DataLoader(
    dataset=Config.train_dataset,
    batch_size=Config.batch_size,
    shuffle=True,
    # num_workers=4,
    pin_memory=True, # 可以加速计算
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=Config.test_dataset,
    batch_size=Config.batch_size,
    shuffle=False,
    # num_workers=1,
    pin_memory=True, # 可以加速计算
)

model = Config.Net().to(device)
optimizer =optim.SGD(model.parameters(),lr=Config.learning_rate,momentum=Config.momentum)
for epoch in range(Config.num_epochs):
    train(model,device,train_dataloader,optimizer,epoch)
test(model,device,test_dataloader)

# torch.save(model.state_dict(),"mnist_cnn.pt")
