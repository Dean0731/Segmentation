import os
from tensorflow import keras
from util import Evaluate,Tools
from network import Model

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from util.dataset.AerialImage import AerialImage
from util.dataset.CountrySide import CountrySide
from util.dataset.Massachusetts import Massachusetts
def selectDataset(str='A',data_size='tif_576',parent="/home/dean/"):
    if str == 'A':
        dataset = AerialImage(parent= parent,data_size=data_size)
    elif str == 'C':
        dataset = CountrySide(parent= parent,data_size=data_size)
    elif str == 'M':
        dataset = Massachusetts(parent= parent,data_size=data_size)
    else:
        print("错误:未找到数据集.")
    return dataset
def complie(model,lr,num_classes):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=[
            Evaluate.MyAccuracy(),
            Evaluate.MyMeanIOU(num_classes=num_classes),
            Evaluate.MyPrecusion(),
            Evaluate.MyRecall(),
            # Evaluate.AveragePrecision
            # Evaluate.P,
            # Evaluate.R,
            # Evaluate.F,
        ]
    )
    return model
def fit(model,data,steps_per_epoch,epochs,validation_data,validation_steps,callbacks):
    model.fit(data,steps_per_epoch=steps_per_epoch,
              validation_data=validation_data,validation_steps=validation_steps,
              epochs=epochs,callbacks=callbacks,
              verbose=1,
              )
    return model


def getNetwork_Model(log=True):
    # 必写参数
    target_size = (576,576)
    mask_size = (576,576)
    num_classes = 2
    batch_size = 3

    # 获取数据
    dataset = selectDataset('C',"{}_{}".format('tif',576),parent='/home/dean/PythonWorkSpace/Segmentation/dataset')
    data,validation_data,test_data = dataset.getData(target_size=target_size,mask_size=mask_size,batch_size=batch_size)

    pre_file = r'h5'
    epochs = 160
    period = max(1,epochs/10) # 每1/10 epochs保存一次
    train_step,val_step,test_step = [dataset.getSize(type)//batch_size for type in ['train','val','test']]


    # 获取模型
    model = Model.getModel('Segnet',target_size,n_labels=2)
    # 是否有与预训练文件，有的话导入
    if os.path.exists(pre_file):
        model.load_weights(pre_file)
    # 生成参数日志文件夹
    if log:
        log_dir,h5_dir,event_dir = Tools.get_dir()
        callback = Evaluate.getCallBack(log_dir,h5_dir,event_dir,period)
    else:
        callback,h5_dir = None,None
    return model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir
@Tools.Decorator.timer(flag=True)
def main():
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = getNetwork_Model()
    model = complie(model,lr=0.001,num_classes=num_classes)
    model = fit(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,callbacks=callback)
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
    model.evaluate(test_data,steps=test_step)


if __name__ == '__main__':
    ret, time = main()
    Tools.sendMessage("The job had cost about {:.2f}小时".format(time//3600))