import os
import tensorflow as tf
from util import Evaluate,Tools,Dataset
from network import Model
import datetime,json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# 自定义，不能用因为，写的不好，OOM异常
def myDefine(model,data,steps_per_epoch,epochs,validation_data,validation_steps,log_dir):
    optimizer = tf.optimizers.Nadam()
    loss_func = tf.losses.CategoricalCrossentropy()

    train_loss = tf.metrics.Mean(name='train_loss')
    train_metric_acc = tf.metrics.CategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.metrics.Mean(name='valid_loss')
    valid_metric_acc = tf.metrics.CategoricalAccuracy(name='valid_accuracy')

    # @tf.function,加上会多占内存
    def train_step(model, features, labels):
        with tf.GradientTape() as tape:
            predictions = model(features,training = True)
            loss = loss_func(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
        train_metric_acc.update_state(labels, predictions)


    # @tf.function
    def valid_step(model, features, labels):
        predictions = model(features)
        batch_loss = loss_func(labels, predictions)
        valid_loss.update_state(batch_loss)
        valid_metric_acc.update_state(labels, predictions)


    for epoch in tf.range(1,epochs+1):
        tf.print("Epoch {}/{}:".format(epoch,epochs))
        step= 1
        while step < steps_per_epoch+1:
            x = next(data[0])
            y = next(data[1])
            train_step(model,x,y)
            tf.print("train - step:{: >3}/{} - {{loss:{:4f},categorical_accuracy:{:4f}}}".format(step,steps_per_epoch,train_loss.result(),train_metric_acc.result()))

        step=1
        while step < validation_steps+1:
            x = next(validation_data[0])
            y = next(validation_data[1])
            valid_step(model,x,y)
            tf.print("val - step:{: >3}/{} - {{loss:{:4f},categorical_accuracy:{:4f}}}".format(step,validation_steps,valid_loss.result(),valid_metric_acc.result()))
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric_acc.reset_states()
        valid_metric_acc.reset_states()



def train_on_batch(model,data,steps_per_epoch,epochs,validation_data,validation_steps,log_dir):
    with open(os.path.join(log_dir,'train.log'),'w',encoding='UTF-8')as f:
        with open(os.path.join(log_dir,'val.log'),'w',encoding="UTF-8")as f2:
            for epoch in tf.range(1,epochs+1):
                tf.print("Epoch {}/{}:".format(epoch,epochs))
                model.reset_metrics()
                # 在后期降低学习率
                if epoch == 5:
                    model.optimizer.lr.assign(model.optimizer.lr/2.0)
                    tf.print("Lowering optimizer Learning Rate...\n\n")
                step= 1
                while step < steps_per_epoch+1:
                    x = next(data[0])
                    y = next(data[1])
                    train_result = model.train_on_batch(x, y)
                    temp_train = dict(zip(model.metrics_names,Tools.getNumbySize(train_result,4)))
                    tf.print("train - step:{: >3}/{} - {}".format(step,steps_per_epoch,temp_train))
                    step = step + 1

                step=1
                while step < validation_steps+1:
                    x = next(validation_data[0])
                    y = next(validation_data[1])
                    valid_result = model.test_on_batch(x, y,reset_metrics=False)
                    temp_val = dict(zip(model.metrics_names,Tools.getNumbySize(valid_result,4)))
                    tf.print("val - step:{: >3}/{} - {}".format(step,validation_steps,temp_val))
                    step = step + 1
                f.write(json.dumps(temp_train)+"\n")
                f2.write(json.dumps(temp_val)+"\n")
                f.flush()
                f2.flush()
    return model

def test_on_batch(model,data,test_steps):
    model.reset_metrics()
    step=1
    while step < test_steps+1:
        x = next(data[0])
        y = next(data[1])
        test_result = model.test_on_batch(x, y,reset_metrics=False)
        step = step + 1
    tf.print("test - step:{: >3}/{} - {}".format(step-1,test_steps,dict(zip(model.metrics_names,Tools.getNumbySize(test_result,4)))))
    return model;


def getNetwork_Model(log=True):
    # 必写参数

    target_size = (512,512)
    mask_size = (512,512)
    num_classes = 2
    batch_size = 2

    # 获取数据
    dataset = Dataset.Dataset(r'G:\AI_dataset\dom\segmentation\data.txt',target_size,mask_size,num_classes)
    data,validation_data,test_data = dataset.getTrainValTestDataset()
    data = data.batch(batch_size)
    validation_data = validation_data.batch(batch_size)

    pre_file = r'h5'
    epochs = 1
    period = max(1,epochs/10) # 每1/10 epochs保存一次
    train_step,val_step,test_step = 3,2,1

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
    tf.print("开始训练".center(20,'*'))
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = getNetwork_Model(log=True)
    model = Evaluate.complie(model,lr=0.001,num_classes=num_classes)
    model = train_on_batch(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,log_dir=h5_dir)
    # model = test_on_batch(model,test_data,test_step)
    tf.print("训练结束".center(20,'*'))

@Tools.Decorator.timer(flag=True)
def main2():
    tf.print("开始训练".center(20,'*'))
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = getNetwork_Model(log=True)
    model = myDefine(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))
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