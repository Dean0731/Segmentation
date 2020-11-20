from pytorch.network.deeplabv3plus import modeling
model = modeling.deeplabv3_resnet50(2, pretrained_backbone=False)
from torchsummary import summary
summary(model,(3,512,512))