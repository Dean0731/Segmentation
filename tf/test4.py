# Pragram:
# 	Segnet 继承Model版本，但init中定义的网络层不能复用，复用会报错
# History:
# 2020-07-06 Dean First Release
# Email:dean07kernel_size1@qq.com
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Lambda,Input
from tensorflow.keras import Model

import tensorflow as tf
def MaxPool2DWithArgmax(input_tensor, ksize, strides):
    p, m = tf.nn.max_pool_with_argmax(input_tensor, ksize=ksize, strides=strides, padding="SAME", include_batch_in_index=True)
    m = tf.cast(m, dtype=tf.int32)
    return [p, m]

def Unpool2D(input_tensors, factor):
    pool, mask = input_tensors
    indices = tf.reshape(mask, (-1,mask.shape[1]*mask.shape[2]*mask.shape[3],1))
    values = tf.reshape(pool, (-1,pool.shape[1]*pool.shape[2]*mask.shape[3]))
    size = tf.size(indices) * factor[1] * factor[2]  # 获取上采样后的数据数量
    size = tf.reshape(size, [-1])  # 转为1维向量，此时里边应该只有一个数
    t = tf.scatter_nd(indices, values, size)
    t = tf.reshape(t, (-1, mask.shape[1]*factor[1], mask.shape[2]*factor[2], mask.shape[3]))  # 恢复四维
    return t

def segnet_simple(height=576,width=576,channel=3,n_labels=2):
    pool_args = {"ksize": (1,2,2,1), "strides":(1,2,2,1)}
    unpool_args = {"factor": (1,2,2,1)}

    inputs = Input((height,width,channel))

    x,mask = Lambda(MaxPool2DWithArgmax, arguments=pool_args)(inputs)
    x,mask = Lambda(MaxPool2DWithArgmax, arguments=pool_args)(x)

    outputs = Activation("softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="Segnet_simple")
if __name__ == '__main__':
    # dropout 也起到正则化效果，但是不如bn，一般只有fc层后边使用，现在几乎不用了，
    # Segnet(512,512,3).summary()

    segnet_simple(512,512,3).summary()