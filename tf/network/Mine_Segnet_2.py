# Pragram:
# 	Segnet网络使用深度可分离卷积代替，参数量缩小约八倍
# History:
# 2020-07-06 Dean First Release
# Email:dean0731@qq.com
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Lambda
import tensorflow as tf


dic = {
    "conv":(64,3),"bn":None,"activate":"relu","conv":(64,3),"bn":None,"activate":"relu","pool":{"ksize": (1,2,2,1), "strides":(1,2,2,1)},
    "conv":(128,3),"bn":None,"activate":"relu","conv":(64,3),"bn":None,"activate":"relu","pool":{"ksize": (1,2,2,1), "strides":(1,2,2,1)},
    "conv":(256,3),"bn":None,"activate":"relu","conv":(64,3),"bn":None,"activate":"relu","pool":{"ksize": (1,2,2,1), "strides":(1,2,2,1)},
    "conv":(512,3),"bn":None,"activate":"relu","conv":(64,3),"bn":None,"activate":"relu","pool":{"ksize": (1,2,2,1), "strides":(1,2,2,1)},
 }
class Segnet(tf.keras.Model):
    def MaxPool2DWithArgmax(self,input_tensor, ksize, strides):
        p, m = tf.nn.max_pool_with_argmax(input_tensor, ksize=ksize, strides=strides, padding="SAME", include_batch_in_index=True)
        m = tf.cast(m, dtype=tf.int32)
        return [p, m]

    def Unpool2D(self,input_tensors, factor):
        pool, mask = input_tensors
        indices = tf.reshape(mask, (-1,mask.shape[1]*mask.shape[2]*mask.shape[3],1))
        values = tf.reshape(pool, (-1,pool.shape[1]*pool.shape[2]*mask.shape[3]))
        size = tf.size(indices) * factor[1] * factor[2]  # 获取上采样后的数据数量
        size = tf.reshape(size, [-1])  # 转为1维向量，此时里边应该只有一个数
        t = tf.scatter_nd(indices, values, size)
        t = tf.reshape(t, (-1, mask.shape[1]*factor[1], mask.shape[2]*factor[2], mask.shape[3]))  # 恢复四维
        return t

    def __init__(self):
        super(Segnet, self).__init__()

        self.bn = BatchNormalization()
        self.activate = Activation('relu')

        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        l = []
        for i in (self.conv1,self.conv2,self.conv3,self.conv4)
            x = self.conv1(x)
            x = self.bn(x)
            x = self.activate(x)
            x, mask_1 = Lambda(self.MaxPool2DWithArgmax, arguments={"ksize": (1,2,2,1), "strides":(1,2,2,1)})(x)
            l.append(mask_1)

        return self.d2(x)
model = Segnet()




def Segnet(height=576,width=576,channel=3,n_labels=2):
    input_shape = (height, width, channel)
    kernel = 3
    args = {"ksize": (1,2,2,1), "strides":(1,2,2,1)}
    pool_size = (1,2,2,1)






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

    return Model(inputs=inputs, outputs=outputs, name="SegNet")
