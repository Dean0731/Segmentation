# @Time     : 2020/7/19 14:19
# @File     : Evaluate
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 网络一些属性配置脚本
# @History  :
#   2020/7/19 Dean First Release
import tensorflow as tf
import os
import json
from util.func import getNumbySize
# train_on_batch
def train_on_batch(model,data,steps_per_epoch,epochs,validation_data,validation_steps,log_dir):
    with open(os.path.join(log_dir,'train.file'),'w',encoding='UTF-8')as f:
        with open(os.path.join(log_dir,'val.file'),'w',encoding="UTF-8")as f2:
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
                    temp_train = dict(zip(model.metrics_names, getNumbySize(train_result, 4)))
                    tf.print("train - epoch:{:>3}/{} - step:{: >3}/{} - {}".format(epoch,epochs,step,steps_per_epoch,temp_train))
                    step = step + 1
                    if step==steps_per_epoch+1:
                        break
                step=1
                for x,y in validation_data:
                    valid_result = model.test_on_batch(x, y,reset_metrics=False)
                    temp_val = dict(zip(model.metrics_names, getNumbySize(valid_result, 4)))
                    tf.print("val - step:{: >3}/{} - {}".format(step,validation_steps,temp_val))
                    step = step + 1
                    if step==validation_steps+1:
                        break
                f.write(json.dumps(temp_train)+"\n")
                f2.write(json.dumps(temp_val)+"\n")
                if epoch%10==0:
                    model.save("epoch_{}.h5".format(epochs))
    return model

def test_on_batch(model,data,test_steps):
    model.reset_metrics()
    step=1
    test_result=None
    for x,y in data:
        test_result = model.test_on_batch(x, y,reset_metrics=False)
        step = step + 1
        if step==test_steps+1:
            break
    tf.print("temp - step:{: >3}/{} - {}".format(step - 1, test_steps, dict(zip(model.metrics_names, func.getNumbySize(test_result, 4)))))
    return model
# 自定义，不能用因为，写的不好，OOM异常
def define_on_train(model,data,steps_per_epoch,epochs,validation_data,validation_steps,log_dir):
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
        for x,y in validation_data:
            valid_step(model,x,y)
            tf.print("val - step:{: >3}/{} - {{loss:{:4f},categorical_accuracy:{:4f}}}".format(step,validation_steps,valid_loss.result(),valid_metric_acc.result()))
            step = step+1
            if step==steps_per_epoch+1:
                break
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric_acc.reset_states()
        valid_metric_acc.reset_states()