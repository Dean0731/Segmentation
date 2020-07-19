import os
from tensorflow import keras
from util import Decorator,Evaluate,GenerateDir,Tools
from util.dataset.AerialImage import AerialImage
from util.dataset.CountrySide import CountrySide
from util.dataset.Massachusetts import Massachusetts
from network import Segnet3,Segnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
def complie(model,lr,num_classes):
    return model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=[Evaluate.MyMeanIOU(num_classes=num_classes), "acc"]
        # metrics=['acc']
    )
def fit(model,data,steps_per_epoch,epochs,validation_data,validation_steps,callbacks):
    model.fit(data,steps_per_epoch=steps_per_epoch,
              validation_data=validation_data,validation_steps=validation_steps,
              epochs=epochs,callbacks=callbacks,# verbose=1,
              )
def selectDataset(str='A',data_size='tif_576'):
    if str == 'A':
        dataset = AerialImage(parent='G:\AI_dataset\法国-房屋数据集2',data_size=data_size)
    elif str == 'C':
        dataset = CountrySide(parent='G:\AI_dataset\DOM',data_size=data_size)
    elif str == 'M':
        dataset = Massachusetts(parent='G:\AI_dataset\马萨诸塞州-房屋数据集1',data_size=data_size)
    else:
        print("错误:未找到数据集.")
    return dataset

@Decorator.timer(flag=True)
def main():
    # 必写参数
    target_size = (576,576)
    mask_size = (352,352)
    num_classes = 2
    batch_size = 2

    # 获取数据
    dataset = selectDataset('M',"{}_{}".format('tif',target_size[0]))
    data,validation_data,test_data = dataset.getData(target_size=target_size,mask_size=mask_size,batch_size=batch_size)
    data.__next__()
    pre_file = r'h5'
    epochs = 160
    period = max(1,epochs/10)
    train_step,val_step,test_step = [dataset.getSize(type)//batch_size for type in ['train','val','test']]

    # 生成参数日志文件夹
    log_dir,h5_dir,event_dir = GenerateDir.get_dir()
    # 获取模型
    model = Segnet.Segnet(target_size[0],target_size[1],3,n_labels=num_classes)
    # model = Unet.Unet(target_width,target_height,3,n_labels=num_classes)
    # 是否有与训练文件，有的话导入
    if os.path.exists(pre_file):
        model.load_weights(pre_file)
    callback = Evaluate.getCallBack(log_dir,h5_dir,event_dir,period)
    complie(model,lr=0.001,num_classes=num_classes)
    fit(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,callbacks=callback)
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
    model.evaluate(test_data,steps=test_step)


if __name__ == '__main__':
    ret, time = main()
    Tools.sendMessage("The job had cost about {:.2f}小时".format(time//3600))