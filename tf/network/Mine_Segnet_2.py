# Pragram:
# 	Segnet 继承Model版本，但init中定义的网络层不能复用，复用会报错
# History:
# 2020-07-06 Dean First Release
# Email:dean07kernel_size1@qq.com
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Lambda
import tensorflow as tf

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
        pool_args = {"ksize": (1,2,2,1), "strides":(1,2,2,1)}
        unpool_args = {"factor": (1,2,2,1)}
        activate = 'relu'
        kernel_size = 3
        self.layer = [
            ("conv",(64,kernel_size)),("bn",None),("activate",activate),("conv",(64,kernel_size)),("bn",None),("activate",activate),("pool",pool_args),
            ("conv2",(128,kernel_size)),("bn",None),("activate",activate),("conv2",(128,kernel_size)),("bn",None),("activate",activate),("pool",pool_args),
            ("conv3",(256,kernel_size)),("bn",None),("activate",activate),("conv3",(256,kernel_size)),("bn",None),("activate",activate),("conv4",(256,1)),("bn",None),("activate",activate),("pool",pool_args),
            ("conv5",(512,kernel_size)),("bn",None),("activate",activate),("conv5",(512,kernel_size)),("bn",None),("activate",activate),("conv6",(512,1)),("bn",None),("activate",activate),("pool",pool_args),
            ("conv5",(512,kernel_size)),("bn",None),("activate",activate),("conv5",(512,kernel_size)),("bn",None),("activate",activate),("conv6",(512,1)),("bn",None),("activate",activate),("pool",pool_args),
            ("unpool",unpool_args), ("conv5",(512,kernel_size)),("bn",None),("activate",activate), ("conv5",(512,kernel_size)),("bn",None),("activate",activate), ("conv5",(512,kernel_size)),("bn",None),("activate",activate),
            ("unpool",unpool_args), ("conv5",(512,kernel_size)),("bn",None),("activate",activate), ("conv5",(512,kernel_size)),("bn",None),("activate",activate), ("conv4",(256,kernel_size)),("bn",None),("activate",activate),
            ("unpool",unpool_args), ("conv3",(256,kernel_size)),("bn",None),("activate",activate), ("conv3",(256,kernel_size)),("bn",None),("activate",activate), ("conv2",(128,kernel_size)),("bn",None),("activate",activate),
            ("unpool",unpool_args), ("conv2",(128,kernel_size)),("bn",None),("activate",activate),("conv",(64,kernel_size)),("bn",None),("activate",activate),
            ("unpool",unpool_args), ("conv",(64,kernel_size)),("bn",None),("activate",activate), ("conv7",(2,1)),("bn",None),("activate",activate),
        ]
        self.conv = Conv2D(64,kernel_size,padding="same")
        self.conv2 = Conv2D(128,kernel_size,padding="same")
        self.conv3 = Conv2D(256,kernel_size,padding="same")
        self.conv4 = Conv2D(256,1,padding="same")
        self.conv5 = Conv2D(512,kernel_size,padding="same")
        self.conv6 = Conv2D(512,1,padding="same")
        self.conv7 = Conv2D(2,1,padding="same")

        self.pool = Lambda(self.MaxPool2DWithArgmax, arguments=pool_args)
        self.unpool = Lambda(self.Unpool2D, arguments=unpool_args)
        self.bn = BatchNormalization()
        self.activate = Activation(activate)
    def call(self, x, training=False):
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
            print(name,args,x.shape)
        return Activation("softmax")(x)
if __name__ == '__main__':
    model = Segnet()
    model.build(input_shape=(4,512,512,3))
    model.summary()
