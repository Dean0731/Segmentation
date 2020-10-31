# @Time     : 2020/7/19 14:19
# @File     : Evaluate
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 网络一些属性配置脚本
# @History  :
#   2020/7/19 Dean First Release
import torch
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
class ToTrainModel():
    def __init__(self,model):
        self.model = model
    def _writer(self,train_log,val_log,epoch):
        if self.writer != None :
            for k,v in train_log.items():
                self.writer.add_scalar(tag=k,scalar_value=v.item(),global_step=epoch)
            for k,v in val_log.items():
                self.writer.add_scalar(tag="val-{}".format(k),scalar_value=v.item(),global_step=epoch)
    def compile(self,loss,optimizer,metrics,device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") or device
        self.optimizer = optimizer
        self.loss = loss
        assert isinstance(metrics,list)
        self.metrics = [self.loss,] + metrics
    def fit(self,data,epochs,val_data,logs_dir=False):
        if os.path.exists(logs_dir):
            self.writer = SummaryWriter(logs_dir)
        else:
            self.writer = None
        for epoch in range(1,epochs+1):
            print("Eopch {}/{}".format(epoch,epochs))
            train_log = ToTrainModel.train(self.model, self.device, data, self.optimizer, epoch,self.loss)
            print("val ....")
            val_log = ToTrainModel.val(self.model, self.device, val_data,self.metrics)
            self._writer(train_log,val_log,epoch)
    @staticmethod
    def train(model,device,train_dataloader,optimizer,epoch,loss_func):
        model.train()
        metrice_dict = {}
        for idx,(data,target) in enumerate(train_dataloader,start=1):
            data,target = data.to(device),target.to(device)
            pred = model(data)['out'] # batch_size * 10
            loss = loss_func(pred,target.long()) # 交叉熵损失函数 ,每个batch中的一张图片的损失
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("Train Epoch:{}, iteration:{}/{}, loss:{}".format(epoch,idx,len(train_dataloader),loss))
        metrice_dict[loss_func.__name__ if hasattr(loss_func,"__name__") else loss_func.__class__.__name__] = loss
        return metrice_dict
    @staticmethod
    def val(model,device,val_dataloader,metrics):
        model.eval()
        metrice_dict={metric.__name__ if hasattr(metric,"__name__") else metric.__class__.__name__ :torch.tensor(0,dtype=torch.float32) for metric in metrics}
        with torch.no_grad():
            for (idx,(data,target)),_ in zip(enumerate(val_dataloader,start=1),tqdm(range(len(val_dataloader)))):
                data,target = data.to(device),target.to(device)
                pred = model(data)['out'] # batch_size * 10
                metrice_dict = ToTrainModel.computerMetrics(metrics,pred,target,metrice_dict)
            metrice_dict = {k:(v/len(val_dataloader.dataset)) for k,v in metrice_dict.items()}
            print("Val - {}".format(metrice_dict))
            return metrice_dict
    def test(self,test_dataloader):
        self.model.eval()
        metrice_dict={metric.__name__ if hasattr(metric,"__name__") else metric.__class__.__name__ :torch.tensor(0,dtype=torch.float32) for metric in self.metrics}
        with torch.no_grad():
            for (idx,(data,target)),_ in zip(enumerate(test_dataloader,start=1),tqdm(range(len(test_dataloader)))):
                data,target = data.to(self.device),target.to(self.device)
                pred = self.model(data)['out'] # batch_size * 10
                metrice_dict = ToTrainModel.computerMetrics(self.metrics,pred,target,metrice_dict)
            metrice_dict = {k:(v/len(test_dataloader.dataset)) for k,v in metrice_dict.items()}
            print("Test - {}".format(metrice_dict))
            return metrice_dict
    def save(self,h5_dir):
        if h5_dir==False:
            return
        elif os.path.exists(h5_dir):
            torch.save(self.model.state_dict(),os.path.join(h5_dir,"last.pt"))
    @staticmethod
    def computerMetrics(metrics,pred,target,metrice_dict):
        for metric in metrics:
            if hasattr(metric,"__name__"):
                metrice_dict[metric.__name__] = metrice_dict[metric.__name__] + metric(pred,target)
            else:
                metrice_dict[metric.__class__.__name__] = metrice_dict[metric.__class__.__name__] + metric(pred,target)
        return metrice_dict