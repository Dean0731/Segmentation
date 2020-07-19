import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential

class BaiscBlcok(layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BaiscBlcok,self).__init__()
        # BasicBlock 第一个卷积
        self.conv1 = layers.Conv2D(filter_num,(3,3),strides=stride,padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation("relu")
        # BasicBlock 第二个卷积，relu并未出现，短接后出现，效果更好
        self.conv2 = layers.Conv2D(filter_num,(3,3),strides=1,padding="same")
        self.bn2 = layers.BatchNormalization()
        # 短接层Identity
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):
        # 卷积1
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 卷积二
        out = self.conv2(out)
        out = self.bn2(out)

        # 短接
        identity = self.downsample(inputs)
        output = layers.add([out,identity])
        output = self.relu(output)
        return output

class ResNet(keras.Model):  # Resnet18
    def __init__(self,layer_dims,num_classes=100):
        # [2,2,2,2]  四个res_block,每个包含2个basicblock
        super(ResNet, self).__init__()

        # 预处理层
        self.stem = Sequential([
            layers.Conv2D(64,(3,3),strides=(1,1)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding="same") # 可以没有
                                ])
        # 中间
        self.layer1 = self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128,layer_dims[1],stride=2) # 降维
        self.layer3 = self.build_resblock(256,layer_dims[2],stride=2)
        self.layer4 = self.build_resblock(512,layer_dims[3],stride=2)

        # 全连接
        self.avgpool = layers.GlobalAvgPool2D()  # 例：假设卷积输出【h，w，c】---》【6,6,3】 ----》【1,1,3】
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None,):

        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b,c]
        x = self.avgpool(x)
        # [b,num_classes]
        x = self.fc(x)

        return x

    def build_resblock(self,filter_num,blocks,stride=1):
        res_blocks = Sequential()

        # 可能有下采样功能
        res_blocks.add(BaiscBlcok(filter_num,stride))

        for _ in range(1,blocks):
            res_blocks.add(BaiscBlcok(filter_num,stride=1))
        return res_blocks

    @staticmethod
    def resnet18():
        return ResNet([2,2,2,2],num_classes=2)

if __name__ == '__main__':
    model = ResNet.resnet18()
    model.build(input_shape=(None,576,576,3))
    model.summary()