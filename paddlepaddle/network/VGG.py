# @Time     : 2020/9/27 17:07
# @File     : VGG
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/9/27 Dean First Release
from paddle import fluid
def vgg_bn_drop(image, type_size):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt, # 具有[N，C，H，W]格式的输入图像
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups, # 过滤器个数
            conv_filter_size=3, # 过滤器大小
            conv_act='relu',
            conv_with_batchnorm=True, # 表示在 Conv2d Layer 之后是否使用 BatchNorm
            conv_batchnorm_drop_rate=dropouts,# 表示 BatchNorm 之后的 Dropout Layer 的丢弃概率
            pool_type='max') # 最大池化

    conv1 = conv_block(image, 64, 2, [0.0, 0])
    conv2 = conv_block(conv1, 128, 2, [0.0, 0])
    conv3 = conv_block(conv2, 256, 3, [0.0, 0.0, 0])
    conv4 = conv_block(conv3, 512, 3, [0.0, 0.0, 0])
    conv5 = conv_block(conv4, 512, 3, [0.0, 0.0, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)

    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.0)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=type_size, act='softmax')
    return predict

def convolutional_neural_network(image, type_size):
    # 第一个卷积--池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,# 输入图像
                                                  filter_size=3,# 滤波器的大小
                                                  num_filters=32,# filter 的数量。它与输出的通道相同
                                                  pool_size=2,# 池化层大小2*2
                                                  pool_stride=2,# 池化层步长
                                                  act='relu') # 激活类型

    # Dropout主要作用是减少过拟合，随机让某些权重不更新
    # Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。
    # 根据给定的丢弃概率dropout随机将一些神经元输出设置为0，其他的仍保持不变。
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)

    # 第二个卷积--池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,
                                                  filter_size=3,
                                                  num_filters=64,
                                                  pool_size=2,
                                                  pool_stride=2,
                                                  act='relu')
    # 减少过拟合，随机让某些权重不更新
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)

    # 第三个卷积--池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop,
                                                  filter_size=3,
                                                  num_filters=64,
                                                  pool_size=2,
                                                  pool_stride=2,
                                                  act='relu')
    # 减少过拟合，随机让某些权重不更新
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    # 全连接层
    fc = fluid.layers.fc(input=drop, size=512, act='relu')
    # 减少过拟合，随机让某些权重不更新
    drop =  fluid.layers.dropout(x=fc, dropout_prob=0.5)
    # 输出层 以softmax为激活函数的全连接输出层，输出层的大小为图像类别type_size个数
    predict = fluid.layers.fc(input=drop,size=type_size,act='softmax')

    return predict