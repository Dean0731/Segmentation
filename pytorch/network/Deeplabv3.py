import torchvision,torch

from torchsummary import summary
deeplabv3_resnet50 = torchvision.models.segmentation.deeplabv3_resnet50

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_resnet50(num_classes=2)
    print(dir(model))
    print(type(model))
    summary(model,(3,512,512))