# @Time     : 2020/7/19 14:19
# @File     : Evaluate
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 网络一些属性配置脚本
# @History  :
#   2020/7/19 Dean First Release
import torch
from pytorch.util import Config

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
            total_iou = total_iou + Config.iou(target,pred).item()
            correct = correct + Config.acc(target,pred).item()
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
            total_iou = total_iou + Config.iou(target,pred).item()
            correct = correct + Config.acc(target,pred).item()
        total_loss = total_loss + Config.loss(pred,target).item()*Config.batch_size
        total_iou = total_iou/len(test_dataloader.dataset)
        total_loss = total_loss/len(test_dataloader.dataset)
        correct = correct/len(test_dataloader.dataset)
        print("Test - loss:{} - acc:{} - miou:{}".format(total_loss,correct,total_iou))
        return "{{loss:{} - acc:{} - miou:{}}}".format(total_loss,correct,total_iou)