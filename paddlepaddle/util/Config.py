import numpy as np
import cv2
from PIL import Image
import paddle
from paddle import fluid
from paddlepaddle.util import Dataset
import matplotlib.pyplot as plt
def transform(sample):
    """
    transform函数的作用是用来对训练集的图像进行处理修剪和数组变换，返回img数组和标签
    sample是一个python元组，里面保存着图片的地址和标签。 ('../images/face/zhangziyi/20181206145348.png', 2)
    """
    img, label = sample
    img = paddle.dataset.image.load_image(img,is_color=True)
    #进行了简单的图像变换，这里对图像进行crop修剪操作，输出img的维度为(3, 100, 100)
    img = cv2.resize(img,target_size,interpolation=cv2.INTER_NEAREST)
    #将img数组进行进行归一化处理，得到0到1之间的数值
    img = img.flatten().astype('float32')/255.0

    label = Image.open(label)
    label = label.resize(mask_size,resample=0)
    label = np.asarray(label)
    return img, label

BATCH_SIZE = 4
target_size = (100,100)
mask_size = (100,100)
num_classes = 2
EPOCH_NUM = 20
# 把图片数据生成reader
dataset = Dataset.Dataset("G:\AI_dataset\dom\segmentation\data_train.txt").getDataset(transform=transform)
# train_reader = paddle.reader.shuffle(reader=dataset,buf_size=300) 可有可无
train_reader = paddle.batch(dataset,batch_size=BATCH_SIZE)
test_reader = paddle.batch(Dataset.Dataset("G:\AI_dataset\dom\segmentation\data_test.txt").getDataset(transform=transform),batch_size=BATCH_SIZE)

# 数据层
image = fluid.layers.data(name='image', shape=(3,)+target_size, dtype='float32')#[3, 100, 100]，表示为三通道，100*100的RGB图
label = fluid.layers.data(name='label', shape=mask_size, dtype='int64')

# 分类器：网络结构
predict = convolutional_neural_network(image=image, type_size=num_classes)

# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label)
# 计算cost中所有元素的平均值
avg_cost = fluid.layers.mean(cost)
#计算准确率
accuracy = fluid.layers.accuracy(input=predict, label=label)

# 定义优化方法
optimizer = fluid.optimizer.Adam(learning_rate=0.001)    # Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计该函数实现了自适应矩估计优化器
optimizer.minimize(avg_cost)  # 取局部最优化的平均损失


# 使用CPU进行训练
place = fluid.CPUPlace
# 创建一个executor
exe = fluid.Executor(place)
# 对program进行参数初始化1.网络模型2.损失函数3.优化函数
exe.run(fluid.default_startup_program())

# 定义输入数据的维度,DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 Executor
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)#定义输入数据的维度，第一个是图片数据，第二个是图片对应的标签。

all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost)
    plt.plot(iters, accs,color='green',label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()
