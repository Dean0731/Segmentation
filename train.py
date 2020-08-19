import os
from util import Evaluate,Tools,Dataset
from network import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

def getNetwork_Model(log=True):
    # 必写参数
    target_size = (576,576)
    mask_size = (576,576)
    num_classes = 2
    batch_size = 2

    # 获取数据
    dataset = Dataset.Dataset(r'G:\AI_dataset\dom\segmentation\data.txt',target_size,mask_size,num_classes)
    data,validation_data,test_data = dataset.getTrainValTestDataset()
    data = data.batch(batch_size)
    validation_data = validation_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    pre_file = r'h5'
    epochs = 1
    period = max(1,epochs/10) # 每1/10 epochs保存一次

    # 获取模型
    model = Model.getModel('segnet',target_size,n_labels=2)
    # 是否有与预训练文件，有的话导入
    if os.path.exists(pre_file):
        model.load_weights(pre_file)
    # 生成参数日志文件夹
    if log:
        log_dir,h5_dir,event_dir = Tools.get_dir()
        callback = Evaluate.getCallBack(log_dir,h5_dir,event_dir,period)
    else:
        callback,h5_dir = None,None
    return model,callback,data,validation_data,test_data,num_classes,epochs,h5_dir
@Tools.Decorator.timer(flag=True)
def main():
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = getNetwork_Model(log=True)
    model = Evaluate.complie(model,lr=0.001,num_classes=num_classes)
    model.fit(data,validation_data=validation_data,epochs=epochs,callbacks=callback)
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
    model.evaluate(test_data)

if __name__ == '__main__':
    msg = ""
    try:
        ret, time = main()
        m, s = divmod(time, 60)
        h, m = divmod(m, 60)
        msg ="The job had cost about {}h{}m{}s".format(h,m,int(s))
    except Exception as error:
        msg = '程序错误，终止！\n{}'.format(error)
    finally:
        if msg==None:
            Tools.sendMessage(msg)
        else:
            Tools.sendMessage("msg为空")