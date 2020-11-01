import numpy as np
import torch
from torchvision import transforms
def toNumpy(x):
    return np.array(x,dtype=np.int)
def jiangwei(x):
    return torch.squeeze(x,dim=0)
def getTargetTransforms(target_size):
    target_transforms = [
        transforms.Resize(target_size,interpolation=0),
        transforms.Lambda(toNumpy),
        transforms.ToTensor(),
        transforms.Lambda(jiangwei)
    ]
    return target_transforms
def getTransforms(input_size):
    input_transform = [
        transforms.Resize(input_size,interpolation=0),
        transforms.ToTensor()
    ]
    return input_transform