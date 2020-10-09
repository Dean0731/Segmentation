# Pragram:
# 	Segnet网络，原始网络
# History:
# 2020-07-06 Dean First Release
# Email:dean0731@qq.com
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Reshape, Lambda, Input, Activation
from tensorflow.keras.models import Model

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
pool_args = {"ksize": (1,2,2,1), "strides":(1,2,2,1)}
unpool_args = {"factor": (1,2,2,1)}
activate = 'relu'
kernel_size = 3
layers = [
    ("conv",(64,kernel_size)),("bn",None),("activate",activate),("conv",(64,kernel_size)),("bn",None),("activate",activate),("pool",pool_args),
    ("conv",(128,kernel_size)),("bn",None),("activate",activate),("conv",(128,kernel_size)),("bn",None),("activate",activate),("pool",pool_args),
    ("conv",(256,kernel_size)),("bn",None),("activate",activate),("conv",(256,kernel_size)),("bn",None),("activate",activate),("conv",(256,1)),("bn",None),("activate",activate),("pool",pool_args),
    ("conv",(512,kernel_size)),("bn",None),("activate",activate),("conv",(512,kernel_size)),("bn",None),("activate",activate),("conv",(512,1)),("bn",None),("activate",activate),("pool",pool_args),
    ("conv",(512,kernel_size)),("bn",None),("activate",activate),("conv",(512,kernel_size)),("bn",None),("activate",activate),("conv",(512,1)),("bn",None),("activate",activate),("pool",pool_args),
    ("unpool",unpool_args), ("conv",(512,kernel_size)),("bn",None),("activate",activate), ("conv",(512,kernel_size)),("bn",None),("activate",activate), ("conv",(512,kernel_size)),("bn",None),("activate",activate),
    ("unpool",unpool_args), ("conv",(512,kernel_size)),("bn",None),("activate",activate), ("conv",(512,kernel_size)),("bn",None),("activate",activate), ("conv",(256,kernel_size)),("bn",None),("activate",activate),
    ("unpool",unpool_args), ("conv",(256,kernel_size)),("bn",None),("activate",activate), ("conv",(256,kernel_size)),("bn",None),("activate",activate), ("conv",(128,kernel_size)),("bn",None),("activate",activate),
    ("unpool",unpool_args), ("conv",(128,kernel_size)),("bn",None),("activate",activate),("conv",(64,kernel_size)),("bn",None),("activate",activate),
    ("unpool",unpool_args), ("conv",(64,kernel_size)),("bn",None),("activate",activate), ("conv",(2,1)),("bn",None),("activate",activate),
]
def segnet_simple(height=576,width=576,channel=3,n_labels=2):
    inputs = Input((height,width,channel))
    x = inputs
    l = []
    for name,args in layers:
        if name == 'conv':
            x = Conv2D(*args,padding="same")(x)
        elif name == 'bn':
            x = BatchNormalization()(x)
        elif name == 'activate':
            x = Activation(args)(x)
        elif name == 'pool':
            x,mask = Lambda(MaxPool2DWithArgmax, arguments=args)(x)
            l.append(mask)
        elif name == 'unpool':
            x = Lambda(Unpool2D, arguments=args)([x, l.pop()])
        else:
            tf.print("错误！{}层不存在".format(name))
    outputs = Activation("softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="Segnet_simple")

def segnet(height=576,width=576,channel=3,n_labels=2):

    # with tf.device('/')
    inputs = Input((height, width, channel))
    x = Conv2D(64, kernel_size, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_1, mask_1 = Lambda(MaxPool2DWithArgmax, arguments=pool_args)(x)

    x = Conv2D(128, kernel_size, padding="same")(pool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_2, mask_2 = Lambda(MaxPool2DWithArgmax, arguments=pool_args)(x)

    x = Conv2D(256, kernel_size, padding="same")(pool_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_3, mask_3 = Lambda(MaxPool2DWithArgmax, arguments=pool_args)(x)

    x = Conv2D(512, kernel_size, padding="same")(pool_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_4, mask_4 = Lambda(MaxPool2DWithArgmax, arguments=pool_args)(x)

    x = Conv2D(512, kernel_size, padding="same")(pool_4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_5, mask_5 = Lambda(MaxPool2DWithArgmax, arguments=pool_args)(x)


    # ======================================================================

    x = Lambda(Unpool2D, arguments=unpool_args)([pool_5, mask_5])

    x = Conv2D(512, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments=unpool_args)([x, mask_4])

    x = Conv2D(512, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments=unpool_args)([x, mask_3])

    x = Conv2D(256, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments=unpool_args)([x, mask_2])

    x = Conv2D(128, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(Unpool2D, arguments=unpool_args)([x, mask_1])

    x = Conv2D(64, kernel_size, padding="same")(x)
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

    return Model(inputs=inputs, outputs=outputs, name="segnet")
class SegnetCls(tf.keras.Model):
    def __init__(self):
        super(SegnetCls, self).__init__()
        self.conv = Conv2D(64,kernel_size,padding="same")
        self.conv2 = Conv2D(128,kernel_size,padding="same")
        self.conv3 = Conv2D(256,kernel_size,padding="same")
        self.conv4 = Conv2D(256,1,padding="same")
        self.conv5 = Conv2D(512,kernel_size,padding="same")
        self.conv6 = Conv2D(512,1,padding="same")
        self.conv7 = Conv2D(2,1,padding="same")

        self.pool = Lambda(MaxPool2DWithArgmax, arguments=pool_args)
        self.unpool = Lambda(Unpool2D, arguments=unpool_args)
        self.bn = BatchNormalization()
        self.activate = Activation(activate)
    def call(self, x, **kwargs):
        l = []
        for name,args in self.layer:
            if name == 'conv':
                x = self.conv(x)
            elif name == 'conv2':
                x = self.conv2(x)
            elif name == 'conv3':
                x = self.conv3(x)
            elif name == 'conv4':
                x = self.conv4(x)
            elif name == 'conv5':
                x = self.conv5(x)
            elif name == 'conv6':
                x = self.conv6(x)
            elif name == 'conv7':
                x = self.conv7(x)
            elif name == 'bn':
                x = self.bn(x)
            elif name == 'activate':
                x = self.activate(x)
            elif name == 'pool':
                x,mask = self.pool(x)
                l.append(mask)
            elif name == 'unpool':
                x = self.unpool([x, l.pop()])
            else:
                tf.print("错误！{}层不存在".format(name))
        return Activation("softmax")(x)


if __name__ == '__main__':
    # dropout 也起到正则化效果，但是不如bn，一般只有fc层后边使用，现在几乎不用了，
    segnet(512,512,3).summary()
    segnet_simple(512,512,3).summary()
