# @Time     : 2020/7/20 20:26
# @File     : Xception
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : Xception
# @History  :
#   2020/7/20 Dean First Release

from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense,SeparableConv2D,MaxPooling2D
from tensorflow.keras.layers import Input,GlobalAveragePooling2D

def _entry_flow(input):
    x = Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64,kernel_size=(3,3),use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # ==================
    residual = Conv2D(filters=128,kernel_size=(1,1),strides=(2,2),padding="same",use_bias=False)(x)
    residual = BatchNormalization()(residual)


    x = SeparableConv2D(filters=128,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(filters=128,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)
    x = layers.add([x,residual])

    residual = Conv2D(filters=256,kernel_size=(1,1),strides=(2,2),padding="same",use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=256,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=256,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)
    x = layers.add([x,residual])


    residual = Conv2D(filters=728,kernel_size=(1,1),strides=(2,2),padding="same",use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)
    x = layers.add([x,residual])
    return x
def _middle_flow(x):
    for i in range(8):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(filters=728,kernel_size=(3,3),padding="same",use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(filters=728,kernel_size=(3,3),padding="same",use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(filters=728,kernel_size=(3,3),padding="same",use_bias=False)(x)
        x = BatchNormalization()(x)

        x = layers.add([x,residual])
    return x
def _exit(x,n_labels):
    residual = Conv2D(filters=1024,kernel_size=(1,1),strides=(2,2),padding="same",use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(filters=1024,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)

    x = layers.add([x,residual])

    x = SeparableConv2D(filters=1536,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(filters=2048,kernel_size=(3,3),padding="same",use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)  # n 个特征图 输出 n个值 变为 1*n向量
    x = Dense(n_labels,activation='softmax')(x)
    return x
def Xception(width=576,height=576,channel=3,n_labels=2):
    input = Input((height, width, channel))
    entry = _entry_flow(input)
    middle = _middle_flow(entry)
    exit = _exit(middle,n_labels)
    model = Model(input,exit)
    return model
if __name__ == '__main__':
    Xception(n_labels=1000).summary()



















