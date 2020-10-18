# @Time     : 2020/7/19 14:19
# @File     : Evaluate
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 网络一些属性配置脚本
# @History  :
#   2020/7/19 Dean First Release
import os
import easydict
import tensorflow as tf
from tensorflow import keras

from util.func import get_dir
from util.cls import DatasetPath

from tf.util.Dataset import Dataset
from tf.network import Model
from util import flag

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
        period = config.period
    )
    def scheduler(epoch):
        if epoch < 20:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    learningRateScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return [tensorBoardDir, checkpoint]

config = easydict.EasyDict()
config.train_model = "segnet"
config.target_size = (512,512)
config.mask_size = (512,512)
config.num_classes = 2
config.log = True
config.EPOCH_NUM = int(flag.get('epochs') or 40)
config.learning_rate = 0.001
config.batch_size = 4
config.pre_file = r'h5'
config.loss = "categorical_crossentropy",
config.period = max(1,config.EPOCH_NUM/10) # 每1/10 epochs保存一次
config.dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.ALL),config.target_size,config.mask_size,config.num_classes)
# data,validation_data,test_data = dataset.getDataset(transform=Transform.transform_double_input)
from tf.util import Transform
config.data,config.validation_data,config.test_data = [dataset.batch(config.batch_size) for dataset in config.dataset.getDataset(transform=Transform.transform_common,seed=7)]
config.model = Model.getModel(config.train_model, config.target_size, n_labels=config.num_classes)
config.model.compile(
    loss=config.loss,
    optimizer=keras.optimizers.Adam(lr=config.learning_rate),
    metrics=[
        tf.metrics.CategoricalAccuracy(),
        MyMeanIOU(num_classes=config.num_classes),
    ]
)
# 是否有与预训练文件，有的话导入
if os.path.exists(config.pre_file):
    config.model.load_weights(config.pre_file)
# 生成参数日志文件夹
if config.log:
    _,h5_dir,event_dir = get_dir()
    config.callback = getCallBack()
    config.h5_dir = h5_dir
else:
    config.callback,config.h5_dir = None,None

if __name__ == '__main__':
    print(DatasetPath('dom').getPath(DatasetPath.ALL))