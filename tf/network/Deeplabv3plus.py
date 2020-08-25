# @Time     : 2020/7/20 19:18
# @File     : Deeplabv3plus
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     : Deeplabv3plus + Xception
#   参考：https://github.com/bubbliiiing/Semantic-Segmentation/tree/master/deeplab_Xception/nets
# @History  :
#   2020/7/20 Dean First Release

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Softmax,Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D,SeparableConv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import backend as K
class Xception:

    def _conv2d_same(self,x, filters,stride=1, kernel_size=3, rate=1,name=""):
        if stride == 1:
            return Conv2D(filters,(kernel_size, kernel_size),strides=(stride, stride),padding='same', use_bias=False,dilation_rate=(rate, rate),name=name)(x)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            return Conv2D(filters,(kernel_size, kernel_size),strides=(stride, stride),padding='valid', use_bias=False,dilation_rate=(rate, rate),name=name)(x)
    def _sepConv_BN(self,x, filters,stride=1, kernel_size=3, rate=1, epsilon=1e-3,name=''):
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),padding=depth_padding, use_bias=False)(x)  # dilation_rate 深度膨胀卷积
        x = BatchNormalization(epsilon=epsilon)(x)
        x = Conv2D(filters, (1, 1), padding='same',use_bias=False)(x)
        x = BatchNormalization(epsilon=epsilon)(x)
        x = Activation('relu')(x)
        return x

    def _entry_flow(self,x):
        x = Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),use_bias=False,padding="same",name="Xception_1")(x)
        x = BatchNormalization(name="Xception_2")(x)
        x = Activation('relu',name="Xception_3")(x)
        x = Conv2D(filters=64,kernel_size=(3,3),use_bias=False,padding="same",name="Xception_4")(x)
        x = BatchNormalization(name="Xception_5")(x)
        x = Activation('relu',name="Xception_6")(x)


        residual = self._conv2d_same(x,128,2,1,name="Xception_7")
        residual = BatchNormalization(name="Xception_8")(residual)
        x = self._sepConv_BN(x,128,stride=1,rate=1,name="Xception_9")
        x = self._sepConv_BN(x,128,stride=1,rate=1,name="Xception_10")
        x = self._sepConv_BN(x,128,stride=2,rate=1,name="Xception_11")
        x = layers.add([x,residual])

        residual = self._conv2d_same(x,256,2,1,name="Xception_12")
        residual = BatchNormalization(name="Xception_13")(residual)
        x = self._sepConv_BN(x,256,stride=1,rate=1,name="Xception_14")
        skip = self._sepConv_BN(x,256,stride=1,rate=1,name="Xception_15")
        x = self._sepConv_BN(skip,256,stride=2,rate=1,name="Xception_16")
        x = layers.add([x,residual])

        residual = self._conv2d_same(x,728,2,1,name="Xception_17")
        residual = BatchNormalization(name="Xception_18")(residual)
        x = self._sepConv_BN(x,728,stride=1,rate=1,name="Xception_19")
        x = self._sepConv_BN(x,728,stride=1,rate=1,name="Xception_20")
        x = self._sepConv_BN(x,728,stride=2,rate=1,name="Xception_21")
        x = layers.add([x,residual])
        return x,skip
    def _middle_flow(self,x):
        for i in range(16):
            residual = x
            shortcut = SepConv_BN(residual,728,stride=1,rate=1,name="Xception_22_{}_1".format(i))
            shortcut = SepConv_BN(shortcut,728,stride=1,rate=1,name="Xception_22_{}_2".format(i))
            shortcut = SepConv_BN(shortcut,728,stride=1,rate=1,name="Xception_22_{}_3".format(i))
            x = layers.add([shortcut,residual])
        return x
    def _exit(self,x):
        residual = self._conv2d_same(x,1024,2,1,name="Xception_23")
        residual = BatchNormalization(name="Xception_24")(residual)
        x = self._sepConv_BN(x,728,stride=1,rate=1,name="Xception_25")
        x = self._sepConv_BN(x,1024,stride=1,rate=1,name="Xception_26")
        x = self._sepConv_BN(x,1024,stride=2,rate=1,name="Xception_27")
        x = layers.add([x,residual])

        x = self._sepConv_BN(x,1536,stride=1,rate=1,name="Xception_28")
        x = self._sepConv_BN(x,1536,stride=1,rate=1,name="Xception_29")
        x = self._sepConv_BN(x,2048,stride=1,rate=1,name="Xception_30")
        return x
    def xception(self,x,OS=16):
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)
        x,skip = self._entry_flow(x)
        x = self._middle_flow(x)
        x = self._exit(x)
        return x,atrous_rates,skip

def SepConv_BN(x, filters,stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3,name=''):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    # 如果需要激活函数
    if not depth_activation:
        x = Activation('relu')(x)

    # 分离卷积，首先3x3分离卷积，再1x1卷积

    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False)(x)  # dilation_rate 深度膨胀卷积
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    x = Conv2D(filters, (1, 1), padding='same',use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    return x

def Deeplabv3(width,height,channel = 3, n_labels=2):
    img_input = Input(shape=(width,height,channel))
    # 主干网络
    x,atrous_rates,skip1 = Xception().xception(img_input,OS=16)

    # ASPP，rate值与Output Strides相关，SepConv_BN为先3x3膨胀卷积，再1x1卷积，进行压缩其膨胀率就是rate值
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, )(x)
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)

    # rate = 6 (12)
    b1 = SepConv_BN(x, 256,rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256,rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256,rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    b4 = GlobalAveragePooling2D()(x)  # 全局池化
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)  # 扩张一维
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)  # 再扩张一维  1*1*channels

    b4 = Conv2D(256, (1, 1), padding='same',use_bias=False)(b4)  # 卷积通道压缩  1*1*256
    b4 = BatchNormalization( epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)

    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.image.resize(x, size_before[1:3]))(b4) # 扩张为64*64*256

    x = Concatenate()([b4,b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)


    x = Lambda(lambda xx: tf.image.resize(xx, skip1.shape[1:3]))(x)


    dec_skip1 = Conv2D(48, (1, 1), padding='same',use_bias=False)(skip1)
    dec_skip1 = BatchNormalization(epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])

    x = SepConv_BN(x, 256,depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256,depth_activation=True, epsilon=1e-5)

    x = Conv2D(n_labels, (1, 1), padding='same')(x)

    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx:tf.image.resize(xx,size_before3[1:3]))(x)

    # x = Reshape((-1,n_labels))(x)
    x = Softmax()(x)

    inputs = img_input
    model = Model(inputs, x)

    return model
if __name__ == '__main__':
    model = Deeplabv3(576,576,3)
    model.summary(line_length=200)