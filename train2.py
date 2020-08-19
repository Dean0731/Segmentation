import os
import tensorflow as tf
from util import Dataset,Config,Tools
import json
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        for x,y in data:
            train_step(model,x,y)
            tf.print("train - step:{: >3}/{} - {{loss:{:4f},categorical_accuracy:{:4f}}}".format(step,steps_per_epoch,train_loss.result(),train_metric_acc.result()))
            step = step+1
            if step==steps_per_epoch+1:
                break
        step=1
        for x,y in data:
            valid_step(model,x,y)
            tf.print("val - step:{: >3}/{} - {{loss:{:4f},categorical_accuracy:{:4f}}}".format(step,validation_steps,valid_loss.result(),valid_metric_acc.result()))
            step = step+1
            if step==steps_per_epoch+1:
                break
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric_acc.reset_states()
        valid_metric_acc.reset_states()


# train_on_batch
def train_on_batch(model,data,steps_per_epoch,epochs,validation_data,validation_steps,log_dir):
    with open(os.path.join(log_dir,'train.log'),'w',encoding='UTF-8')as f:
        with open(os.path.join(log_dir,'val.log'),'w',encoding="UTF-8")as f2:
            for epoch in tf.range(1,epochs+1):
                tf.print("Epoch {}/{}:".format(epoch,epochs))
                model.reset_metrics()
                # 在后期降低学习率
                # if epoch == 5:
                #     model.optimizer.lr.assign(model.optimizer.lr/2.0)
                #     tf.print("Lowering optimizer Learning Rate...\n\n")
                step= 1
                for x,y in data:
                    train_result = model.train_on_batch(x, y)
                    temp_train = dict(zip(model.metrics_names,Tools.getNumbySize(train_result,4)))
                    tf.print("train - epoch:{:>3}/{} - step:{: >3}/{} - {}".format(epoch,epochs,step,steps_per_epoch,temp_train))
                    step = step + 1
                    if step==steps_per_epoch+1:
                        break
                step=1
                for x,y in validation_data:
                    valid_result = model.test_on_batch(x, y,reset_metrics=False)
                    temp_val = dict(zip(model.metrics_names,Tools.getNumbySize(valid_result,4)))
                    tf.print("val - step:{: >3}/{} - {}".format(step,validation_steps,temp_val))
                    step = step + 1
                    if step==validation_steps+1:
                        break
                f.write(json.dumps(temp_train)+"\n")
                f2.write(json.dumps(temp_val)+"\n")
                f.flush()
                f2.flush()
                if epoch%10==0:
                    model.save_weight("epoch_{}.h5".format(epochs))
    return model

def test_on_batch(model,data,test_steps):
    model.reset_metrics()
    step=1
    for x,y in data:
        test_result = model.test_on_batch(x, y,reset_metrics=False)
        step = step + 1
        if step==test_steps+1:
            break;
    tf.print("test - step:{: >3}/{} - {}".format(step-1,test_steps,dict(zip(model.metrics_names,Tools.getNumbySize(test_result,4)))))
    return model;

# model.fit  compile --->train
@Tools.Decorator.timer(flag=True)
def main():
    # 获取数据
    target_size = (512,512)
    mask_size = (512,512)
    num_classes = 2
    batch_size = 2
    # dataset = Dataset.Dataset(r'G:\AI_dataset\dom\segmentation\data.txt',target_size,mask_size,num_classes)
    dataset = Dataset.Dataset(r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation/data.txt',target_size,mask_size,num_classes)
    #dataset = Dataset.CountrySide(r'/home/dean/PythonWorkSpace/Segmentation/dataset/dom/segmentation/data.txt',target_size,mask_size,num_classes)

    tf.print("开始训练".center(20,'*'))
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = Config.getNetwork_Model("mysegnet_4", dataset, batch_size, target_size, num_classes)
    model = Config.complie(model, lr=0.001, num_classes=num_classes)
    model.fit(data, validation_data=validation_data,steps_per_epoch=train_step,validation_steps=val_step,epochs=epochs,callbacks=callback,verbose=1)
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
    model.evaluate(test_data,steps=test_step)
    tf.print("训练结束".center(20,'*'))

# model.train_on_batch compile    ----> train（自己可以控制）
@Tools.Decorator.timer(flag=True)
def main1():
    # 获取数据
    target_size = (512,512)
    mask_size = (512,512)
    num_classes = 2
    batch_size = 2
    # dataset = Dataset.Dataset(r'G:\AI_dataset\dom\segmentation\data.txt',target_size,mask_size,num_classes)
    dataset = Dataset.Dataset(r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation/data.txt',target_size,mask_size,num_classes)
    #dataset = Dataset.CountrySide(r'/home/dean/PythonWorkSpace/Segmentation/dataset/dom/segmentation/data.txt',target_size,mask_size,num_classes)
    #dataset = Dataset.CountrySide(r'E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation\data.txt',target_size,mask_size,num_classes)

    tf.print("开始训练".center(20,'*'))
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = Config.getNetwork_Model("mysegnet_4", dataset, batch_size, target_size, num_classes)
    model = Config.complie(model, lr=0.001, num_classes=num_classes)
    model = train_on_batch(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model = test_on_batch(model,test_data,test_step)

# 自定义 compile 自定义train
@Tools.Decorator.timer(flag=True)
def main2():
    # 获取数据
    target_size = (512,512)
    mask_size = (512,512)
    num_classes = 2
    batch_size = 2
    # dataset = Dataset.Dataset(r'G:\AI_dataset\dom\segmentation\data.txt',target_size,mask_size,num_classes)
    dataset = Dataset.Dataset(r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation/data.txt',target_size,mask_size,num_classes)
    #dataset = Dataset.CountrySide(r'/home/dean/PythonWorkSpace/Segmentation/dataset/dom/segmentation/data.txt',target_size,mask_size,num_classes)

    tf.print("开始训练".center(20,'*'))
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = Config.getNetwork_Model("mysegnet_4", dataset, batch_size, target_size, num_classes)
    model = myDefine(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))
if __name__ == '__main__':
    ret, time = main2()
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    msg ="The job had cost about {}h{}m{}s".format(h,m,int(s))
    Tools.sendMessage(msg)