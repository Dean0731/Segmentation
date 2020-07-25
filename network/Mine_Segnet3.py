# Pragram:
# 	Segnet网络使用深度可分离卷积代替，参数量缩小约八倍
# History:
# 2020-07-06 Dean First Release
# Email:dean0731@qq.com
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Activation,DepthwiseConv2D,MaxPooling2D,UpSampling2D,Concatenate,ZeroPadding2D
from tensorflow.keras.models import Model
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
def aspp(x):
    atrous_rates = (6,12,12)
    b0 = Conv2D(32, (1, 1), padding='same', use_bias=False,strides=2)(x)
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    b1 = SepConv_BN(x, 32,rate=atrous_rates[0], stride=2,depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 32,rate=atrous_rates[1], stride=2,depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 32,rate=atrous_rates[2], stride=2,depth_activation=True, epsilon=1e-5)
    x = Concatenate()([b0, b1, b2, b3])
    return x
def Segnet(height=576,width=576,channel=3,n_labels=2):
    input_shape = (height, width, channel)
    kernel = 3
    kernel_conv = 1
    pool_size = (2,2)

    inputs = Input(input_shape)
    x = aspp(inputs)
    x = aspp(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(64, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(64, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_1 = MaxPooling2D(pool_size=pool_size)(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(pool_1)
    x = Conv2D(128, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(128, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_2 = MaxPooling2D(pool_size=pool_size)(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(pool_2)
    x = Conv2D(256, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(256, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(256, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_3 = MaxPooling2D(pool_size=pool_size)(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(pool_3)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_4 = MaxPooling2D(pool_size=pool_size)(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(pool_4)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_5 = MaxPooling2D(pool_size=pool_size)(x)


    # ======================================================================

    x = UpSampling2D()(pool_5)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512,(kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(512, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(256,(kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(256, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(256, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(128,(kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(128, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(64, (kernel_conv, kernel_conv), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(x)
    x = Conv2D(64,(kernel_conv, kernel_conv), padding="same")(x)
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

    return Model(inputs=inputs, outputs=outputs, name="SegNet 深度可分离网络")
if __name__ == '__main__':
    # dropout 也起到正则化效果，但是不如bn，一般只有fc层后边使用，现在几乎不用了，
    Segnet(3072,3072,3).summary(line_length=200)
