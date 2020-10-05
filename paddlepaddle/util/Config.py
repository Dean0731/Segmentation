import numpy as np
import paddle
from PIL import Image


from paddlepaddle.util.Dataset import Dataset
from paddlepaddle.network import Segnet
from util.cls import DatasetPath

# device = paddle.set_device('gpu')
# paddle.disable_static(device)
paddle.disable_static()

import paddle.nn.functional as F
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
    """
    损失函数
    """
    def __init__(self):
        super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, input, label):
        loss = F.softmax_with_cross_entropy(input,
                                            label,
                                            return_softmax=False,
                                            axis=1)
        return paddle.mean(loss)

BATCH_SIZE = 2
target_size = (512,512)
mask_size = (512, 512)
num_classes = 2
EPOCH_NUM = 20
log_dir='source/test'
train_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TRAIN),transform=transpose)
val_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.VAL),transform=transpose)
test_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TEST),transform=transpose)
model = paddle.Model(Segnet.UNet(num_classes))
