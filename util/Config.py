# @Time     : 2020/7/19 14:19
# @File     : Evaluate
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 网络一些属性配置脚本
# @History  :
#   2020/7/19 Dean First Release
import os,json
import tensorflow as tf
from tensorflow import keras
from util import Tools,Dataset
from network import Model
class Path:
    Shiyanshi_benji = r'E:\Workspace\PythonWorkSpace\Segmentation\dataset\dom\segmentation2\data.txt'
    Shiyanshi_hu= r'/home/dean/PythonWorkSpace/Segmentation/dataset/dom/segmentation2/data.txt'
    lENOVO_PC = r'G:\AI_dataset\dom\segmentation2\data.txt'
    Chaosuan = r'/public1/data/weiht/dzf/workspace/Segmentation/dataset/dom/segmentation2/data.txt'

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
class MyPrecusion(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
class MyRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

def complie(model,lr,num_classes):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=[
            tf.metrics.CategoricalAccuracy(),
            MyMeanIOU(num_classes=num_classes),
            MyPrecusion(),
            MyRecall(),
        ]
    )
    return model

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
    tf.print("test - step:{: >3}/{} - {}".format(step-1,test_steps,dict(zip(model.metrics_names,Tools.getNumbySize(test_result,4)))))
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

def getCallBack(h5_dir, event_dir, period):

    tensorBoardDir = keras.callbacks.TensorBoard(log_dir=event_dir)
    checkpoint = keras.callbacks.ModelCheckpoint(
        # 保存路径
        # h5_dir,
        os.path.join(h5_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
        monitor='val_loss',
        # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
        mode='auto',
        # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型,检测值有改进时保存
        save_best_only=True,
        # 只保存权重文件
        save_weights_only=True,
        # 每个checkpoint之间epochs间隔数量
        period = period
    )
    def scheduler(epoch):
        if epoch < 20:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    learningRateScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return [tensorBoardDir, checkpoint]

def getNetwork_Model(model,target_size,mask_size,num_classes,log=True):
    # 必写参数
    learning_rate = 0.001
    batch_size = 2
    data_txt_path = Path.Shiyanshi_hu
    # 获取数据
    if str(model).startswith("mysegnet"):
        dataset = Dataset.CountrySide(data_txt_path,target_size,mask_size,num_classes)
    else:
        dataset = Dataset.Dataset(data_txt_path,target_size,mask_size,num_classes)
    data,validation_data,test_data = dataset.getTrainValTestDataset()
    data = data.batch(batch_size)
    validation_data = validation_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    pre_file = r'h5'
    epochs= 80
    period = max(1,epochs/10) # 每1/10 epochs保存一次
    # 获取模型,与数据集
    model = Model.getModel(model,target_size,n_labels=num_classes)
    # 是否有与预训练文件，有的话导入
    if os.path.exists(pre_file):
        model.load_weights(pre_file)
    # 生成参数日志文件夹
    if log:
        log_dir,h5_dir,event_dir = Tools.get_dir()
        callback = getCallBack(h5_dir,event_dir,period)
    else:
        callback,h5_dir = None,None
    return model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir

