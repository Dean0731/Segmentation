# 图片分割
 - 语义分割：图片中的房子都是都是房子
    - 实例分割：图片中的房子是房子但是是不同的房子
# 图像分类
每类图片对应一个文件夹
# 目标检测
每张图片对应一个xml或txt，里面对应多个对象信息，一个对象（类别，四个坐标）
# 评价指标
 - IOU:多分类，验证集中每张图片中n各类别，求出每个图片中某一类的iou，做平均得到iou_class1，iou = [iou_class1,iou_class1...]
 - MIOU:将IOU除以类别数即可
# 论文参考
## 基础网络
|名称|时间|论文|详细|
|---|---|---|---|
|FNC||| |
|Unet|2015.5.18|https://arxiv.org/pdf/1505.04597.pdf|
|segnet|2016.10|https://arxiv.org/pdf/1511.00561.pdf|  |
|DeepMask|2015.11|https://arxiv.org/abs/1506.06204   | |
|ParseNet|	2015.11.9|||
|LSTM-CF|		2016|||
|ReSeg	|	2016.5|||
|deeplab|		2016.6|||
|ENet	|	2016.7|||
|refineNet| 	2016.11	|	https://arxiv.org/pdf/1611.06612v3.pdf||
|PSPnet		|2017.4|||
|GCN		    2017|||
|<font color=red>北京大学提出的大分辨率图像,图像分割Semantic segmentation of high-resolution images </font>|2017| http://scis.scichina.com/en/2017/123101.pdf|modified joint  bilateral upsampling algorithm|
|mask-rcnn	|2017.5|||
|deeplabv2  | 2017.5	|	https://arxiv.org/pdf/1606.00915.pdf||
|deeplabv3	|2017.7		|https://arxiv.org/pdf/1706.05587.pdf||
|ICENT，适合高分图像|2018.08.20|https://arxiv.org/pdf/1704.08545.pdf||
|deeplabv3+	|2018.8.22	|	https://arxiv.org/pdf/1802.02611.pdf|https://github.com/tensorflow/models/tree/master/research/deeplab|
|Fast SCNN|2019.02.12|https://arxiv.org/pdf/1902.04502.pdf||
|hrnet|2019.02.25|https://arxiv.org/pdf/1902.09212.pdf||
|Gated-SCNN	|2019.7.12	|	https://arxiv.org/pdf/1907.05740v1.pdf||
|Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation|2020.07.17|https://arxiv.org/pdf/2007.09183.pdf|https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch|


## 2，学习率   
    supervised learning base method  
    region method   
    hierarchical features
# 优化方向
    梯度下降优化函数    
    修改损失函数  
    所建模型    
	    2，后边添加自定义块，例如树形结构   
	    3，修改网络形状，直线型，U行，可以用原型   
	    4，最新的金字塔卷积  
    精度提升    
    速度提升    
	减少参数量:卷积方式
		空洞卷积：增大感受野，但参数不变
		深度可分离：减少了参数 2016.10 - 2017.4
# 问题：
    问题：高分辨率图片，3000*3000 包含全部建筑物，3000*3000  Segnet 输出参数量约50GB     
# 方法：
    1，高像素图片：深度可分离+大卷积核，在网络前加上处理结构     
    2，结尾添加结构，准确率
    3，重定义网络
# 泛化性能增加
| 方法         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| 使用更多数据 | 在有条件的前提下，尽可能多地获取训练数据是最理想的方法，更多的数据可以让模型得到充分的学习，也更容易提高泛化能力 |
| 使用更大批次 | 在相同迭代次数和学习率的条件下，每批次采用更多的数据将有助于模型更好的学习到正确的模式，模型输出结果也会更加稳定 |
| 调整数据分布 | 大多数场景下的数据分布是不均匀的，模型过多地学习某类数据容易导致其输出结果偏向于该类型的数据，此时通过调整输入的数据分布可以一定程度提高泛化能力 |
| 调整目标函数 | 在某些情况下，目标函数的选择会影响模型的泛化能力，如目标函数在某类样本已经识别较为准确而其他样本误差较大的侵害概况下，不同类别在计算损失结果的时候距离权重是相同的，若将目标函数改成则可以使误差小的样本计算损失的梯度比误差大的样本更小，进而有效地平衡样本作用，提高模型泛化能力 |
| 调整网络结构 | 在浅层卷积神经网络中，参数量较少往往使模型的泛化能力不足而导致欠拟合，此时通过叠加卷积层可以有效地增加网络参数，提高模型表达能力；在深层卷积网络中，若没有充足的训练数据则容易导致模型过拟合，此时通过简化网络结构减少卷积层数可以起到提高模型泛化能力的作用 |
| 数据增强     | 数据增强又叫数据增广，在有限数据的前提下通过平移、旋转、加噪声等一些列变换来增加训练数据，同类数据的表现形式也变得更多样，有助于模型提高泛化能力，需要注意的是数据变化应尽可能不破坏元数数据的主体特征(如在图像分类任务中对图像进行裁剪时不能将分类主体目标裁出边界)。 |
| 权值正则化   | 权值正则化就是通常意义上的正则化，一般是在损失函数中添加一项权重矩阵的正则项作为惩罚项，用来惩罚损失值较小时网络权重过大的情况，此时往往是网络权值过拟合了数据样本(如)。 |
| 屏蔽网络节点 | 该方法可以认为是网络结构上的正则化，通过随机性地屏蔽某些神经元的输出让剩余激活的神经元作用，可以使模型的容错性更强。 |
# 内存占用 
    图片缩放---》576 Segnet 2GB  
    图片卷积---》768 Segnet 5GB  
    图片卷积---》384 Segnet 2GB  
# 云端IDE
 - 华为云 https://console.huaweicloud.com/modelarts/ 适合学习，小数据集 ，大数据集无法上传
   - tf2.1 pytorch 1.0 cuda10.2
 - 阿里云 https://dsw-dev.data.aliyun.com/ 可学习用，可训练大数据及集，但不能联网，数据集最好是tar.gz格式  
   - cuda10 环境版本可更新，但通常更新后无法使用
 - 百度云 https://aistudio.baidu.com/aistudio/projectdetail/194452 也可训练大数据集
   - cuda 10.1
   
# 损失函数
 - 最后一层卷积层，就是网络的结束部分，后面的层没有权重参数，就不属于网络结构
 - 损失函数，决定梯度是否和继续迭代，决定了是否有梯度消失现象
 - loss，就是预测值与真实值之间的差，用数字形式表示出来
 - 网络最后输出形状为 (n,h,w,num_lables)
  - 最后使用激活函数softmax或sigmod(n,h,w,mnum_lables)，
  - softmax 将第四维度转化为num_lables 个数，加起来为1
  - sigmod 转化为num_lables个数，但是相加不等于1
 - 分类损失函数
   - 交叉熵损失， 一般结合softmax，$y_1 logx_1 + y_2 logx_2+...+y_n log x_n$
# 实验结论
  网络使用预先训练过的，有预训练参数，准确度会上升约为1%-1.5%
  损失函数会影响准确率，使用bce+dice loss与交叉熵相比增加 有小幅度的提高约为0.1%