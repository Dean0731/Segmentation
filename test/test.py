import tensorflow as tf
from util import Evaluate
import numpy as np

def a():
    img = [
                [0,0,0,0],
                [0,1,1,0],
                [0,1,1,0],
                [0,0,0,0],
        ],

    label = [
                [0,0,0,0],
                [0,0,1,1],
                [0,0,1,1],
                [0,0,0,0],
            ],
    img = np.asarray(img).reshape(4,4)
    label = np.asarray(label).reshape(4,4)
    # img = img.swapaxes(0,2)
    # label = label.swapaxes(0,2)
    return img,label

def b():
    img = [
        [
            [
                [0,0,0,0],
                [0,1,1,0],
                [0,1,1,0],
                [0,0,0,0],
            ],
            [
                [1,1,1,1],
                [1,0,0,1],
                [1,0,0,1],
                [1,1,1,1],
            ]
        ]
    ]
    label = [
         [
            [
                [0,0,0,0],
                [0,0,1,1],
                [0,0,1,1],
                [0,0,0,0],
            ],
            [
                [1,1,1,1],
                [1,1,0,0],
                [1,1,0,0],
                [1,1,1,1],
            ],
        ]
    ]
    img = np.asarray(img)
    label = np.asarray(label)
    img = img.swapaxes(1,3)
    img = img.swapaxes(1,2)
    label = label.swapaxes(1,3)
    label = label.swapaxes(1,2)
    return img,label

def metrics(img,label):
    m = tf.keras.metrics.Accuracy()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)
    m = tf.keras.metrics.Precision()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)
    m = tf.keras.metrics.Recall()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)
def metrics2(img,label):

    m = tf.keras.metrics.TruePositives()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)
    m = tf.keras.metrics.TrueNegatives()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)
    m = tf.keras.metrics.FalsePositives()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)
    m = tf.keras.metrics.FalseNegatives()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)

def metrics3(img,label):

    m = Evaluate.MyAccuracy()
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)

    m = Evaluate.MyMeanIOU(num_classes=2)
    _ = m.update_state(img,label)
    res = m.result().numpy()
    print(res)

    # m = tf.keras.metrics.Precision()
    # _ = m.update_state(tf.argmin(img,axis=-1),tf.argmin(label,axis=-1))
    # res = m.result().numpy()
    # print(res)
    # m = tf.keras.metrics.Recall()
    # _ = m.update_state(tf.argmin(img,axis=-1),tf.argmin(label,axis=-1))
    # res = m.result().numpy()
    m = Evaluate.Recall(img,label)
    print(m)
    m = Evaluate.Precision(img,label)
    print(m)
    m = Evaluate.F1(img,label)
    print(m)
    m = Evaluate.AveragePrecision(img,label)
    print(m)
if __name__ == '__main__':

    # img,label = a()
    # print(img,label)
    # print(img.shape,label.shape)
    # metrics(img,label)

    # print(img.shape,label.shape)
    # print(img)
    # print(label)
    # metrics(img,label)


    img2,label2 = b()
    print(img2[0,:,:,0].shape,label2[0,:,:,0].shape)
    print(img2[0,:,:,0],label2[0,:,:,0])
    img2 = tf.constant(img2,dtype=tf.int64)
    label2 = tf.constant(label2,dtype=tf.int64)
    metrics3(img2[0,:,:,:],label2[0,:,:,:])

    # img = img.reshape(4,4,2)
    # label = label.reshape(4,4,2)
    # print(img.shape,label.shape)
    #
    # img = tf.constant(img,dtype=tf.float32)
    # label = tf.constant(label,dtype=tf.float32)
    # metrics3(img,label)

    # img,label = tf.argmin(img[0,:,:,:], axis=-1), tf.argmin(label[0,:,:,:], axis=-1)
    # metrics(img,label)
    # metrics2(img,label)

