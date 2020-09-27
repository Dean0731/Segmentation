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

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
class MyPrecusion(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
class MyRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
def my_loss(y_true,y_pred):
    pass
def complie(model,lr,num_classes):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=[
            tf.metrics.CategoricalAccuracy(),
            MyMeanIOU(num_classes=num_classes),
        ]
    )
    return model
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
# 数据集转换 普通转换
def transform_common(line_x,line_y):
    image = tf.io.read_file(line_x)
    image = tf.image.decode_png(image)
    image = tf.image.resize(image,target_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = image

    label = tf.io.read_file(line_y)
    label = tf.image.decode_png(label,channels=1)
    label = tf.image.resize(label,mask_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.squeeze(label)
    label = tf.cast(label,dtype=tf.uint8)
    label = tf.one_hot(label,depth=2)   # 注意在进行one-hot前 原数组 的书应是【0,1,2,3】  不能是 【0,2,6】 中间不能空
    label = label[:,:,0]
    label = tf.cast(label,dtype=tf.uint8)
    label = tf.one_hot(label,depth=2)
    y = label
    return x,y

# 双输入
def transform_double_input(line_x,line_y):
    image = tf.io.read_file(line_x)
    image = tf.image.decode_png(image)
    image_1 = tf.image.resize(image,target_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_2 = tf.image.resize(image,(3072,3072),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x_1,x_2 = image_1,image_2

    label = tf.io.read_file(line_y)
    label = tf.image.decode_png(label,channels=1)
    label = tf.image.resize(label,mask_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.squeeze(label)
    label = tf.cast(label,dtype=tf.uint8)
    label = tf.one_hot(label,depth=2)
    label = label[:,:,0]
    label = tf.cast(label,dtype=tf.uint8)
    label = tf.one_hot(label,depth=2)
    y = label
    return (x_1,x_2),y

train_model = "segnet"
target_size = (512,512)
mask_size = (512,512)
num_classes = 2
log = True
EPOCH_NUM = 20
def getNetwork_Model():
    # 必写参数
    learning_rate = 0.001
    batch_size = 2
    data_txt_path = DatasetPath('dom').getPath()
    # 获取数据
    dataset = Dataset(DatasetPath("dom").getPath(),target_size,mask_size,num_classes)

    # data,validation_data,test_data = dataset.getDataset(transform=transform_double_input)
    data,validation_data,test_data = dataset.getDataset(transform=transform_common,seed=7)

    data = data.batch(batch_size)
    validation_data = validation_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    pre_file = r'h5'
    epochs= 80
    period = max(1,epochs/10) # 每1/10 epochs保存一次
    # 获取模型,与数据集
    model = Model.getModel(train_model, target_size, n_labels=num_classes)
    # 是否有与预训练文件，有的话导入
    if os.path.exists(pre_file):
        model.load_weights(pre_file)
    # 生成参数日志文件夹
    if log:
        log_dir,h5_dir,event_dir = get_dir()
        callback = getCallBack(h5_dir,event_dir,period)
    else:
        callback,h5_dir = None,None
    return model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir,num_classes
if __name__ == '__main__':
    print(DatasetPath('dom').getPath())