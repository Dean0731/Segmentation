from pytorch.network.deeplabv3plus.backbone import resnet
model = resnet.__dict__['resnet50'](
    pretrained=False,
    replace_stride_with_dilation=[False, True, True])
from torchsummary import summary
summary(model,(3,512,512))