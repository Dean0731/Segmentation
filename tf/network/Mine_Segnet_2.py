# Pragram:
# 	Segnet网络使用深度可分离卷积代替，参数量缩小约八倍
# History:
# 2020-07-06 Dean First Release
# Email:dean0731@qq.com
from tensorflow.keras.layers import Conv2D,Lambda, BatchNormalization, Input, Activation,DepthwiseConv2D,MaxPooling2D,UpSampling2D,Concatenate,ZeroPadding2D
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K
def MaxPool2DWithArgmax(input_tensor, ksize, strides):
    p, m = tf.nn.max_pool_with_argmax(input_tensor, ksize=ksize, strides=strides, padding="SAME", include_batch_in_index=True)
    m = K.cast(m, dtype=tf.int32)
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

def SepConv_BN(x, filters,stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3,name=''):
    # 如果需要激活函数
    if not depth_activation:
        x = Activation('relu')(x)

    # 分离卷积，首先3x3分离卷积，再1x1卷积

    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding='same', use_bias=False)(x)  # dilation_rate 深度膨胀卷积

    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    x = Conv2D(filters, (1, 1), padding='same',use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    return x
def aspp(x,atrous_rates = [6,12,18]):
    stride = 2
    b0 = Conv2D(32,(3,3),padding='same',strides=stride)(x)
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    b1 = SepConv_BN(x, 32,rate=atrous_rates[0], stride=stride,depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 32,rate=atrous_rates[1], stride=stride,depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 32,rate=atrous_rates[2], stride=stride,depth_activation=True, epsilon=1e-5)
    x = Concatenate()([b0, b1, b2,b3])
    return x

def Segnet(height=576,width=576,channel=3,n_labels=2):
    input_shape = (height, width, channel)
    kernel = 3
    args = {"ksize": (1,2,2,1), "strides":(1,2,2,1)}
    pool_size = (1,2,2,1)

    inputs = Input(input_shape)
    x = aspp(inputs,[4,8,12])
    x = aspp(x,[4,8,12])
    x = Conv2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_1, mask_1 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

    x = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_2, mask_2 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

    x = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_3, mask_3 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

    x = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_4, mask_4 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)

    x = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_5, mask_5 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)


    # ======================================================================

    x = Lambda(Unpool2D, arguments={"factor": pool_size})([pool_5, mask_5])

    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_4])

    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_3])

    x = Conv2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_2])

    x = Conv2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments={"factor": pool_size})([x, mask_1])

    x = Conv2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_labels, (1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Reshape(
    #     (input_shape[0] * input_shape[1], n_labels),
    #     input_shape=(input_shape[0], input_shape[1], n_labels),
    # )(x)

    outputs = Activation("softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="mine_segnet")
if __name__ == '__main__':
    # dropout 也起到正则化效果，但是不如bn，一般只有fc层后边使用，现在几乎不用了，
    Segnet(768,768,3).summary()
