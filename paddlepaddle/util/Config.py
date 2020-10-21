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
class SoftmaxWithCrossEntropy(paddle.nn.Layer):
    def __init__(self):
        super(SoftmaxWithCrossEntropy, self).__init__()
    def forward(self, input, label):
        # 将softmax 与loss结合，
        loss = F.softmax_with_cross_entropy(input,
                                            label,
                                            return_softmax=False,
                                            axis=1)
        return paddle.mean(loss)
class MyAcc(paddle.metric.Accuracy):
    def compute(self, pred, label, *args):
        acc = []
        for x,y in zip(pred,label) :
            x = paddle.argmax(x,axis=0)
            res = paddle.cast(paddle.equal(x,y),dtype="float32")
            acc.append(paddle.reduce_mean(res))
        return paddle.tensor.stack(x=acc)
class MeanIOU(paddle.metric.Accuracy):
    def __init__(self, topk=(1, ), name='iou',num_classes = 2, *args, **kwargs):
        super(MeanIOU, self).__init__(topk=(1, ), name=name, *args, **kwargs)
        self.num_classes = num_classes
    def compute(self, pred, label, *args):
        pred = paddle.argmax(pred,1)
        iou1 = []
        iou2 = []
        for i in range(len(pred)):
            mean_iou, out_wrong, out_correct = paddle.metric.mean_iou(pred[i],label[i],2)
            iou1.append(out_correct[0]/(out_wrong[0]+out_correct[0]))
            iou2.append(out_correct[1]/(out_wrong[1]+out_correct[1]))
        iou1 = paddle.tensor.stack(iou1)
        iou2 = paddle.tensor.stack(iou2)
        iou1 = paddle.mean(iou1)
        iou2 = paddle.mean(iou2)
        return paddle.squeeze(paddle.tensor.stack([iou1,iou2]),axis=1)
    def update(self, correct, *args):
        self.result = correct
        return self.result
    def accumulate(self):
        return self.result

BATCH_SIZE = 8
target_size = (512,512)
mask_size = (512, 512)
num_classes = 2
EPOCH_NUM = int(flag.get('epoch') or 40)
learning_rate = 0.0001
log_dir='~/source/paddlepaddle/'
loss = SoftmaxWithCrossEntropy()
metrics = [MyAcc(name='acc'),MeanIOU(name='iou')]
callback = [
    Visual(log_dir=log_dir),
    paddle.callbacks.ModelCheckpoint(save_freq=5,save_dir=os.path.join(log_dir,"checkpoint")),
    paddle.callbacks.ProgBarLogger(log_freq=1,verbose=2)
]
train_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TRAIN),transform=transpose)
val_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.VAL),transform=transpose)
test_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TEST),transform=transpose)
model = paddle.Model(Unet.UNet(num_classes))
