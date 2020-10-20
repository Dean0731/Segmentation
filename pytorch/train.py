# @Time     : 2020/8/26 17:20
# @File     : train
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release

import torch,os
import torch.optim as optim
from pytorch.util import Config
from pytorch.util import TrainMethod
import util
print("Pytorch Version",torch.__version__)
def main1():
    model = Config.model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    with open(os.path.join(Config.event_dir,'train.txt'),'w',encoding='UTF-8') as f:
        for epoch in range(Config.num_epochs):
            train_log = TrainMethod.train(model, Config.device, Config.train_dataloader, optimizer, epoch)
            val_log = TrainMethod.val(model, Config.device, Config.val_dataloader)
            f.write("{};{}\n".format(train_log,val_log))
        test_log = TrainMethod.test(model, Config.device, Config.test_dataloader)
        f.write(test_log)
    torch.save(model.state_dict(),os.path.join(Config.h5_dir,"mnist_cnn.pt"))

@util.cls.Decorator.sendMessageWeChat()
@util.cls.Decorator.timer()
def main():
    main1()

if __name__ == '__main__':
    main()
