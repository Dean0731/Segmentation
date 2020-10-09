import numpy as np
import paddle
from PIL import Image
from paddlepaddle.util.Dataset import Dataset
from paddlepaddle.network import Unet
from util.cls import DatasetPath
device = paddle.set_device(paddle.device.get_device())
paddle.disable_static(device)
print("使用设备：",device)
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

BATCH_SIZE = 8
target_size = (512,512)
mask_size = (512, 512)
num_classes = 2
EPOCH_NUM = 20
learning_rate = 0.001
log_dir='source/test'
train_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TRAIN),transform=transpose)
val_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.VAL),transform=transpose)
test_dataset = Dataset(DatasetPath("dom").getPath(DatasetPath.TEST),transform=transpose)
model = paddle.Model(Unet.UNet(num_classes))
