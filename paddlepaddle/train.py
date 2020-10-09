# @Time     : 2020/9/27 16:54
# @File     : train
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/9/27 Dean First Release


from paddlepaddle.util.Config import *
import util
@util.cls.Decorator.sendMessageWeChat()
@util.cls.Decorator.timer(flag=True)
def main():
    optim = paddle.optimizer.RMSProp(learning_rate=learning_rate,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False,parameters=model.parameters())
    model.prepare(optim, SoftmaxWithCrossEntropy())
    model.fit(train_dataset,val_dataset,save_dir=log_dir,epochs=EPOCH_NUM,batch_size=BATCH_SIZE,verbose=1)
    model.evaluate(test_dataset,)
if __name__ == '__main__':
    main()