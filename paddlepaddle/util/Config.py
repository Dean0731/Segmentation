import numpy as np
import paddle
import os
import logging
from PIL import Image
import paddle.nn.functional as F

from paddlepaddle.util.Dataset import Dataset
from paddlepaddle.util.Callback import Visual
from paddlepaddle.network import Unet
from util.cls import DatasetPath
from util import flag

device = paddle.set_device(paddle.device.get_device())
paddle.disable_static(device)
logging.info("使用设备：{}".format(device))

def transpose(image,mode='image'):
    image = Image.open(image)
    if mode == 'image':
        image = image.resize(target_size, Image.NEAREST)
        img = np.array(image, dtype='float32')
    else:
        image = image.resize(mask_size, Image.NEAREST)
        img = np.array(image, dtype='uint8')
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

    return img.transpose((2,0,1))
def changeDimAndReshape(input,label):
    input = paddle.flatten(input,2,-1)
    label = paddle.flatten(label,2,-1)
    input = paddle.transpose(input,perm=(0,2,1))
    label = paddle.transpose(label,perm=(0,2,1))
    return input,label
class CrossEntropy(paddle.nn.Layer):
    """
    损失函数
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
    def forward(self, input, label):
        input,label = changeDimAndReshape(input,label)
        sum = []
        for i in range(len(input)):
            sum.append(F.cross_entropy(input[i],label[i],reduction='mean'))
        return paddle.tensor.stack(x=sum)
class MyAcc(paddle.metric.Accuracy):
    def compute(self, pred, label, *args):
        acc = []
        for x,y in zip(pred,label) :
            x = paddle.argmax(x,axis=0)
            res = paddle.cast(paddle.equal(x,y),dtype="float32")
            acc.append(paddle.reduce_mean(res))
        return paddle.tensor.stack(x=acc)
class MeanIOU(paddle.metric.Metric):
    def __init__(self,name=None, *args, **kwargs):
        super(MeanIOU, self).__init__(*args, **kwargs)
        self._name = [name or 'miou']
        self.reset()

    def compute(self, pred, label, *args):
        mean_iou, out_wrong, out_correct = paddle.metric.mean_iou(pred,label,2)
        return mean_iou

    def update(self, correct, *args):
        return correct

    def reset(self):
        self.total = [0.] * len(self.topk)
        self.count = [0] * len(self.topk)

    def accumulate(self):
        res = []
        for t, c in zip(self.total, self.count):
            r = float(t) / c if c > 0 else 0.
            res.append(r)
        res = res[0] if len(self.topk) == 1 else res
        return res


    def name(self):
        """
        Return name of metric instance.
        """
        return self._name

BATCH_SIZE = 8
target_size = (512,512)
mask_size = (512, 512)
num_classes = 2
EPOCH_NUM = int(flag.get('epoch') or 40)
learning_rate = 0.001
log_dir='source/paddlepaddle/'
loss = CrossEntropy()
metrics = [MyAcc(name='acc')]
callback = [
    Visual(log_dir=log_dir),
    paddle.callbacks.ModelCheckpoint(save_freq=5,save_dir=os.path.join(log_dir,"checkpoint")),
    paddle.callbacks.ProgBarLogger(log_freq=1,verbose=2)
]
train_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TRAIN),transform=transpose)
val_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.VAL),transform=transpose)
test_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TEST),transform=transpose)
model = paddle.Model(Unet.UNet(num_classes))
