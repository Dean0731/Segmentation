# @Time     : 2020/8/26 17:20
# @File     : train
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release

import torch
import torch.optim as optim
from pytorch.util import Config
from util import Tools
print("Pytorch Version",torch.__version__)
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
def train(model,device,train_dataloader,optimizer,epoch):
    model.train()
    for idx,(data,target) in enumerate(train_dataloader):
        data,target = data.to(device),target.to(device)
        pred = model(data) # batch_size * 10
        loss = Config.loss(pred,target.long()) # 交叉熵损失函数 ,每个batch中的一张图片的损失
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Train Epoch:{}, iteration:{}, loss:{}".format(epoch,idx,loss))
    return "{{loss:{}}}".format(loss)
def val(model,device,val_dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total_iou = 0
    with torch.no_grad():
        for idx,(data,target) in enumerate(val_dataloader):
            data,target = data.to(device),target.to(device)
            pred = model(data) # batch_size * 10
            total_iou = total_iou + iou(target,pred).item()
            correct = correct + acc(target,pred).item()
            total_loss = total_loss + Config.loss(pred,target).item()*Config.batch_size
        total_iou = total_iou/(len(val_dataloader.dataset))
        total_loss = total_loss/len(val_dataloader.dataset)
        correct = correct/len(val_dataloader.dataset)
        print("Val - loss:{} - acc:{} - miou:{}".format(total_loss,correct,total_iou))
        return "{{loss:{} - acc:{} - miou:{}}}".format(total_loss,correct,total_iou)

def test(model,device,test_dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total_iou = 0
    with torch.no_grad():
        for idx,(data,target) in enumerate(test_dataloader):
            data,target = data.to(device),target.to(device)
            pred = model(data) # batch_size * 10
            total_iou = total_iou + iou(target,pred).item()
            correct = correct + acc(target,pred).item()
        total_loss = total_loss + Config.loss(pred,target).item()*Config.batch_size
        total_iou = total_iou/len(test_dataloader.dataset)
        total_loss = total_loss/len(test_dataloader.dataset)
        correct = correct/len(test_dataloader.dataset)
        print("Test - loss:{} - acc:{} - miou:{}".format(total_loss,correct,total_iou))
        return "{{loss:{} - acc:{} - miou:{}}}".format(total_loss,correct,total_iou)

@Tools.Decorator.sendMessage()
@Tools.Decorator.timer()
def main():
    model = Config.model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    with open('train.txt','w',encoding='UTF-8') as f:
        for epoch in range(Config.num_epochs):
            train_log = train(model, Config.device, Config.train_dataloader, optimizer, epoch)
            val_log = val(model, Config.device, Config.val_dataloader)
            f.write("{};{}\n".format(train_log,val_log))
        test_log = test(model, Config.device, Config.test_dataloader)
        f.write(test_log)

# torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    main()
