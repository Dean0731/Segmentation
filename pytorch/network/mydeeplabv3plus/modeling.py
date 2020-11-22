from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3,ASPP
from .backbone import resnet
from torch import nn
class MyDeep(nn.Module):
    def __init__(self,front, backbone, classifier):
        super(MyDeep, self).__init__()
        self.front = front
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.front(x)
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    front = ASPP(3, aspp_dilate)
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,input_channel=64)
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(2048,256, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = MyDeep(front,backbone, classifier)
    return model
model = _segm_resnet('deeplabv3plus', 'resnet50', 2, output_stride=8, pretrained_backbone=False)


