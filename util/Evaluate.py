# @Time     : 2020/7/19 14:19
# @File     : Evaluate
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : 工具脚本
# @History  :
#   2020/7/19 Dean First Release
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

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

def getCallBack(log_dir, h5_dir, event_dir, period):

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