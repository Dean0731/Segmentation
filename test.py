from tensorflow.keras import Model,layers
import  tensorflow as tf
class SAEModel(Model):
    def __init__(self):
        super(SAEModel, self).__init__()

        # 初始化模型使用的layer，layer_1为前述自定义layer
        # layer_2为全连接层，采用sigmoid激活函数
        # 每层在这里可以不考虑输入元素个数，但必须考虑输出元素个数
        # 输入元素个数可以在call()函数中动态确定
        self.layer_2 = layers.Dense(10)
        # self.layer_3 = layers.Dense(10)
        # self.layer_4 = layers.Dense(10)
        # self.layer_5 = layers.Dense(10)
        # self.layer_5 = layers.Dense(10)


    def call(self, input_tensor, training=False):
        # 输入数据
        x = self.layer_2(input_tensor)
        x = self.layer_2(x)
        x = self.layer_2(x)
        output = self.layer_2(x)
        return output
m = SAEModel()
m.build((4,10))
m.summary()
