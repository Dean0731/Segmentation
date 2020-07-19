# Pragram:
# 	Segnet网络使用深度可分离卷积代替，参数量缩小约八倍
# History:
# 2020-07-06 Dean First Release
# Email:dean0731@qq.com
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Activation,DepthwiseConv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Model
def Segnet(height=576,width=576,channel=3,n_labels=2):
    input_shape = (height, width, channel)
    kernel = 3
    kernel_conv = 1
    pool_size = (2,2)

    inputs = Input(input_shape)

    x = DepthwiseConv2D((kernel, kernel), padding="same")(inputs)
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
    Segnet(576,576,3).summary()
