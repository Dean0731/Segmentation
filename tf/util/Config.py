# @Time     : 2020/7/19 14:19
# @File     : Evaluate
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 网络一些属性配置脚本
# @History  :
#   2020/7/19 Dean First Release
import os
import tensorflow as tf
from tensorflow import keras
from util.func import get_dir
from util.cls import DatasetPath
from tf.util.Dataset import Dataset
from tf.network import Model
from util import flag
import util
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred,axis=-1)
        y_pred = tf.reshape(y_pred,shape=(-1,512*512))
        y_true = tf.reshape(y_true,shape=(-1,512*512))
        super().update_state(y_true,y_pred,sample_weight)


class MyAcc(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred,axis=-1)
        y_pred = tf.reshape(y_pred,shape=(-1,512*512))
        y_true = tf.reshape(y_true,shape=(-1,512*512))
        super().update_state(y_true,y_pred,sample_weight)

def getCallBack():

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

train_model = "deeplabv3plus"
input_size = (512,512)
target_size = (512,512)
num_classes = 2
log = True
num_epochs = int(flag.get('epochs') or 40)
learning_rate = 0.001
batch_size = 2
pre_file = r'h5'
path = DatasetPath("dom")
loss = tf.keras.losses.SparseCategoricalCrossentropy()
# loss = tf.keras.losses.CategoricalCrossentropy()
period = max(1,num_epochs/10) # 每1/10 epochs保存一次
dataset = Dataset(path.getPath(DatasetPath.ALL),input_size,target_size,num_classes)
# data,validation_data,test_data = dataset.getDataset(transform=Transform.transform_double_input)
from tf.util import Transform
data,validation_data,test_data = [dataset.batch(batch_size) for dataset in dataset.getDataset(transform=Transform.transform_common,seed=7)]
model = Model.getModel(train_model, target_size, n_labels=num_classes)
optimizer = keras.optimizers.Adam(lr=learning_rate)
metrics=[
    MyAcc(),
    MyMeanIOU(num_classes=num_classes)

]
model.compile(loss=loss,optimizer= optimizer,metrics=metrics)

# 是否有与预训练文件，有的话导入
if os.path.exists(pre_file):
    model.load_weights(pre_file)
log = int(flag.get('log'))

if log:
    log_,h5_dir,event_dir = get_dir(os.path.join(util.getParentDir(),'source/tensorflow'))
    callback = getCallBack()
    with open(os.path.join(log_,"txt"),"w")as f:
        f.write("metrice:{}\n".format(metrics))
        f.write("input_size:{}\n".format(input_size))
        f.write("target_size:{}\n".format(target_size))
        f.write("in_channels:3\n")
        f.write("out_channels:{}\n".format(num_classes))
        f.write("batch_size:{}\n".format(batch_size))
        f.write("datasetPath:{}\n".format(path))
        f.write("learningRate:{}\n".format(learning_rate))
        f.write("num_epochs:{}\n".format(num_epochs))
        f.write("device:{}\n".format('GUP' if tf.test.is_gpu_available() else "cup"))
        f.write("model:{}\n".format(model.name))
        f.write("optimizer:{}\n".format(optimizer))
        f.write("loss:{}\n".format(loss))
else:
    h5_dir,event_dir,callback = False,False,None

if __name__ == '__main__':
    print(DatasetPath('dom').getPath(DatasetPath.ALL))






