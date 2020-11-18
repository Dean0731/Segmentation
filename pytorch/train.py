# @Time     : 2020/8/26 17:20
# @File     : train
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release

import torch
from pytorch.util import Config
from pytorch.util import TrainMethod
print("Pytorch Version",torch.__version__)
def main1():
    pass
def main2():
    model = TrainMethod.ToTrainModel(Config.model)
    model.compile(Config.loss,Config.optimizer,Config.metrics)
    model.fit(Config.train_dataloader,Config.num_epochs,Config.val_dataloader,logs_dir=Config.log)
    model.test(test_dataloader=Config.test_dataloader)
    model.save(Config.log)
def main():
    main2()

if __name__ == '__main__':
    main()
