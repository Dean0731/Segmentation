# Pragram:
# 	Segnet网络，双输入网络模型，一个输入是大分辨率直接缩放到288，一个输入是3000分辨率，经过卷积然后与另一个连接 再经过Segnet
#   输入1:512*512  ---> 卷积256
#   输入2:3072*3072 --->11*11 3---->11*11 1
# History:
# 2020-08-10 Dean First Release
# Email:dean0731@qq.com
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Reshape, Lambda, Input, Activation,MaxPool2D,Concatenate,Add
from tensorflow.keras.models import Model

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
def input_other(height=3072,width=3072,channel=3,n_labels=2):
    input = Input((height,width,channel))
    x = Conv2D(filters=32,kernel_size=(11,11),strides=(3,3),padding="same",name="conv_other_1")(input)
    # x = Conv2D(filters=32,kernel_size=(11,11),strides=(1,1),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64,kernel_size=(11,11),strides=(2,2),padding="same",name="conv_other_2")(x)
    # x = Conv2D(filters=64,kernel_size=(11,11),strides=(1,1),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128,kernel_size=(11,11),strides=(2,2),padding="same",name="conv_other_3")(x)
    # x = Conv2D(filters=128,kernel_size=(11,11),strides=(1,1),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128,kernel_size=(11,11),strides=(2,2),padding="same",name="conv_other_4")(x)
    # x = Conv2D(filters=128,kernel_size=(11,11),strides=(1,1),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return Model(inputs=input,outputs=x)

def Segnet(height=576,width=576,channel=3,n_labels=2):
    input_shape = (height, width, channel)
    kernel = 3
    args = {"ksize": (1,2,2,1), "strides":(1,2,2,1)}
    pool_size = (1,2,2,1)

    inputs = Input(input_shape)

    x = Conv2D(64, (kernel, kernel), padding="same",name="conv_1_1")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (kernel, kernel), padding="same",name="conv_1_2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    pool_1, mask_1 = Lambda(MaxPool2DWithArgmax, arguments=args)(x)


    x = Conv2D(128, (kernel, kernel), padding="same",name="conv_2_1")(pool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (kernel, kernel), padding="same",name="conv_2_2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    model_A = Model(inputs=inputs,outputs=x)
    model_B = input_other()
    # combined = Concatenate()([model_A.output,model_B.output])
    combined = Add()([model_A.output,model_B.output])
    x = combined
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

    return Model(inputs=[model_A.input,model_B.input], outputs=outputs, name="MySegNet_4")
if __name__ == '__main__':
    # dropout 也起到正则化效果，但是不如bn，一般只有fc层后边使用，现在几乎不用了，
    # Segnet(512,512,3).summary()
    Segnet(256,256,3).summary()
    pass
