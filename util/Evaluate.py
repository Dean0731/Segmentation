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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
def getCallBack(log_dir, h5_dir, event_dir, period):

    tensorBoardDir = keras.callbacks.TensorBoard(log_dir=event_dir)
    checkpoint = keras.callbacks.ModelCheckpoint(
        # 保存路径
        # h5_dir,
        os.path.join(h5_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
        monitor='val_acc',
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

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
class MyAccuracy(tf.keras.metrics.Accuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
class MyPrecusion(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
class MyRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


# average = [None,'micro','macro','weighted','samples']
# average = average[3]
# # average = average[3] # 这两个都可以 map会不一样
#
# def Precision(y_true, y_pred):
#     y_true,y_pred = tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)
#     return precision_score(y_true, y_pred,average=average)
# def Recall(y_true, y_pred):
#     y_true,y_pred = tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)
#     return recall_score(y_true, y_pred,average=average)
# def F1(y_true, y_pred):
#     y_true,y_pred = tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)
#     return f1_score(y_true, y_pred,average=average)
# def AveragePrecision(y_true, y_pred):
#     y_true,y_pred = tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)
#     return average_precision_score(y_true, y_pred,average=average)

from tensorflow.keras import backend as K
def P(y_true, y_pred):
    y_true,y_pred = K.cast(K.argmax(y_true,axis=-1),'float32'),K.cast(K.argmax(y_pred,axis=-1),'float32')
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall
def R(y_true, y_pred):
    y_true,y_pred = K.cast(K.argmax(y_true,axis=-1),'float32'),K.cast(K.argmax(y_pred,axis=-1),'float32')
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))
    recall = true_positives / (poss_positives + K.epsilon())
    return recall

#f-measure
def F(y_true, y_pred):
    p_val = P(y_true, y_pred)
    r_val = R(y_true, y_pred)
    f_val = 2*p_val*r_val / (p_val + r_val)

    return f_val