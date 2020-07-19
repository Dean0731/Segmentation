import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
# 卷积核数量
filter=64
# 卷积核大小
kernel_size=3
# 卷积步长
strides=2
# 方式
padding="same"
# 偏执
use_bias=False
# 激活函数
activation="relu"
# 池化大小
pool_size = 2
pool_strides = 2
pool_padding = "same"
class_num =2
def make_generator_model():

    model = tf.keras.models.Sequential()
    # 1
    model.add(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_strides,padding=pool_padding))
    # 2
    model.add(tf.keras.layers.Conv2D(filters=filter*2, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter*2, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding))
    # 3
    model.add(tf.keras.layers.Conv2D(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding))
    # 4
    model.add(tf.keras.layers.Conv2D(filters=filter * 8, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter * 8, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding))
    # 5
    model.add(tf.keras.layers.Conv2D(filters=filter * 16, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter * 16, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding))
    # start
    model.add(tf.keras.layers.Conv2DTranspose(filters=filter*8,kernel_size=kernel_size,strides=strides,padding=padding,use_bias=use_bias))
    # 拼接
    model.add(tf.keras.layers.concatenate([],axis=1))
    model.add(tf.keras.layers.Conv2D(filters=filter * 8, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter * 8, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    #
    model.add(tf.keras.layers.Conv2DTranspose(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    # 拼接
    model.add(tf.keras.layers.Conv2D(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    #
    model.add(tf.keras.layers.Conv2DTranspose(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    # 拼接
    model.add(tf.keras.layers.Conv2D(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter * 4, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    #
    model.add(tf.keras.layers.Conv2DTranspose(filters=filter * 2, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    # 拼接
    model.add(tf.keras.layers.Conv2D(filters=filter * 2, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter * 2, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    #
    model.add(tf.keras.layers.Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    # 拼接
    model.add(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=use_bias))
    model.add(tf.keras.layers.Activation(activation))

    model.add(tf.keras.layers.Conv2D(filters=class_num, kernel_size=1, strides=1, padding=padding,use_bias=use_bias))
    return model
model =make_generator_model()
model.build((None, 3000, 3000,3))
model.summary()
