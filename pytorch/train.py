# @Time     : 2020/8/26 17:20
# @File     : train
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release

import torch,os

from pytorch.util import Config
from pytorch.util import TrainMethod
import util
print("Pytorch Version",torch.__version__)
def main1():
    with open(os.path.join(Config.event_dir,'train.txt'),'w',encoding='UTF-8') as f:
        for epoch in range(Config.num_epochs):
            train_log = TrainMethod.train(Config.model, Config.device, Config.train_dataloader, Config.optimizer, epoch)
            val_log = TrainMethod.val(Config.model, Config.device, Config.val_dataloader)
            f.write("{};{}\n".format(train_log,val_log))
        test_log = TrainMethod.test(Config.model, Config.device, Config.test_dataloader)
        f.write(test_log)
    torch.save(Config.model.state_dict(),os.path.join(Config.h5_dir,"mnist_cnn.pt"))

def main2():
    model = TrainMethod.ToTrainModel(Config.model)
    model.compile(Config.loss,Config.optimizer,Config.metrics)
    model.fit(Config.train_dataloader,Config.num_epochs,Config.val_dataloader,logs_dir=Config.event_dir)
    model.test(test_dataloader=Config.test_dataloader)
    model.save(Config.h5_dir)
@util.cls.Decorator.sendMessageWeChat()
@util.cls.Decorator.timer()
def main():
    main2()

if __name__ == '__main__':
    main()
