import os
from tensorflow import keras
import tensorflow as tf
from util import Evaluate,Tools
from util.dataset import dataset_tools
from network import Model
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


def fit2(model,data,steps_per_epoch,epochs,validation_data,validation_steps,callbacks):
    x = data[0][0]  # 576
    x1 = data[1][0] # 3072
    y = data[0][1]  # 576
    val_x = validation_data[0][0]  # 576
    val_x1 = validation_data[1][0] # 3072
    val_y = validation_data[0][1]  # 576

    acc_meter = Evaluate.keras.metrics.Accuracy()
    op = tf.keras.optimizers.Adam(0.01)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            x_img = [x.__next__(),x1.__next__()]
            y_label = y.__next__()

            with tf.GradientTape() as tape:
                loss = tf.losses.categorical_crossentropy(y_label,model([x_img]))
                # loss = tf.losses.binary_crossentropy(y_label,model([x_img]))
            grads = tape.gradient(loss,model.trainable_variables) # 求梯度
            op.apply_gradients(zip(grads,model.trainable_variables)) # 更新梯度 w = w - delta
            print("epochs:{}/{},step:{}/{},loss:{:5f}".format(epoch,epochs,step,steps_per_epoch,tf.reduce_mean(loss).numpy()))   # numpy() 将tensor转化为变量
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
            Evaluate.F,
        ]
    )
    return model

def fit(model,data,steps_per_epoch,epochs,validation_data,validation_steps,callbacks):
    print(type(data[0]))
    print(type(data[1]))
    model.fit(x=data[0],y=data[1],steps_per_epoch=steps_per_epoch,
              # validation_data=validation_data,validation_steps=validation_steps,
              epochs=epochs,callbacks=callbacks,
              #verbose=1,
              )
    return model

def getNetwork_Model(log=True):
    # 必写参数

    target_size = (512,512)
    mask_size = (512,512)
    num_classes = 2
    batch_size = 2

    # 获取数据
    # dataset = selectDataset('C',"{}_{}".format('tif',3072),parent='/home/dean/PythonWorkSpace/segmentation/dataset')
    dataset = dataset_tools.selectDataset('C2',"{}_{}".format('tif',3072),parent='/home/dean/PythonWorkSpace/Segmentation/dataset')

    data,validation_data,test_data = dataset.getData(target_size=target_size,mask_size=mask_size,batch_size=batch_size)


    pre_file = r'h5'
    epochs = 160
    period = max(1,epochs/10) # 每1/10 epochs保存一次
    # train_step,val_step,test_step = [dataset.getSize(type)//batch_size for type in ['train','val','test']]
    train_step,val_step,test_step = 500,26,26


    # 获取模型
    model = Model.getModel('mysegnet_4',target_size,n_labels=2)
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
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = getNetwork_Model(log=True)
    model = complie(model,lr=0.001,num_classes=num_classes)
    model = fit(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,callbacks=callback)


if __name__ == '__main__':
    ret, time = main()
    # try:
    #     ret, time = main()
    #     msg ="The job had cost about {:.2f}小时".format(time//3600)
    # except Exception as error:
    #     msg = '程序错误，终止！\n{}'.format(error)
    # finally:
    #     Tools.sendMessage(msg)
