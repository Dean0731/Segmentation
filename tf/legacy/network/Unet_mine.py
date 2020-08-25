# Unet.py
# author:Dean
# tf:2.0
# desc: 加入了BatchNormalization
from tf.keras.layers import Conv2D, BatchNormalization, Lambda, Input, Activation,MaxPooling2D,UpSampling2D,Concatenate
from tf.keras.models import Model
def Unet(height=576,width=576,channel=3,n_labels=2):
    input_shape = (height, width, channel)
    kernel = 3
    pool_size = (2,2)
    pool_strides = (2,2)
    uppool_size = (2,2)

    inputs = Input(input_shape)
    x = Conv2D(64, (kernel, kernel), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_1,conv1 = MaxPooling2D(pool_size=pool_size,strides=pool_strides,padding="same")(x),x

    x = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_2,conv2 = MaxPooling2D(pool_size=pool_size,strides=pool_strides,padding="same")(x),x

    x = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_3,conv3 = MaxPooling2D(pool_size=pool_size,strides=pool_strides,padding="same")(x),x

    x = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_4,conv4 = MaxPooling2D(pool_size=pool_size,strides=pool_strides,padding="same")(x),x

    x = Conv2D(1024, (kernel, kernel), padding="same")(pool_4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(1024, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # ======================================================================

    x = UpSampling2D(size=uppool_size)(x)
    x = Concatenate(axis=3)([x,conv4])

    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=uppool_size)(x)
    x = Concatenate(axis=3)([x,conv3])

    x = Conv2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=uppool_size)(x)
    x = Concatenate(axis=3)([x,conv2])

    x = Conv2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=uppool_size)(x)
    x = Concatenate(axis=3)([x,conv1])

    x = Conv2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_labels, (1, 1), padding="valid")(x)
    x = BatchNormalization()(x)


    outputs = Activation("softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="SegNet")
if __name__ == '__main__':
    # dropout 也起到正则化效果，但是不如bn，一般只有fc层后边使用，现在几乎不用了，
    Unet(576,576,3).summary()
