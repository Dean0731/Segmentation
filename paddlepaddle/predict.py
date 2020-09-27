# @Time     : 2020/9/27 17:00
# @File     : predict
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/9/27 Dean First Release
# coding:utf-8
import paddle.fluid as fluid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import paddle

# 使用CPU进行训练
place = fluid.CPUPlace()
# 定义一个executor
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()#要想运行一个网络，需要指明它运行所在的域，确切的说： exe.Run(&scope)
#选择保存不同的训练模型
params_dirname ="/home/aistudio/data/model_cnn"
#params_dirname ='/home/aistudio/data/model_vgg'

# （1）图片预处理
def load_image(path):
    img = paddle.dataset.image.load_and_transform(path,100,100, False).astype('float32')#img.shape是(3, 100, 100)
    img = img / 255.0
    return img

infer_imgs = []
infer_path = []
zzy = '/home/aistudio/images/face/zhangziyi/20181206144436.png'
jw = '/home/aistudio/images/face/pengyuyan/20181206161115.png'
pyy = '/home/aistudio/images/face/jiangwen/0acb8d12-f929-11e8-ac67-005056c00008.jpg'
dzf = 'images/face/dzf/timg.jpg'
infer_path.append((Image.open(zzy), load_image(zzy)))
infer_path.append((Image.open(jw), load_image(jw)))
infer_path.append((Image.open(pyy), load_image(pyy)))
infer_path.append((Image.open(dzf), load_image(dzf)))
print('infer_imgs的维度：',np.array(infer_path[0][1]).shape)

#fluid.scope_guard修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope
with fluid.scope_guard(inference_scope):
    #获取训练好的模型
    #从指定目录中加载 推理model(inference model)
    [inference_program,# 预测用的program
     feed_target_names,# 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
     fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。

    image_and_path = infer_path[3]
    plt.imshow(image_and_path[0])   #根据数组绘制图像
    plt.show()        #显示图像

    # 开始预测
    results = infer_exe.run(
        inference_program,                      #运行预测程序
        feed={feed_target_names[0]: np.array([image_and_path[1]])},#喂入要预测的数据
        fetch_list=fetch_targets)               #得到推测结果
    print('results:',np.argmax(results[0]))

    # 训练数据的标签
    label_list = ["zhangziyi","jiangwen","pengyuyan","dzf"]
    print(results)
    print("infer results: %s" % label_list[np.argmax(results[0])])