import torch 
import torch.nn as nn 
from torchvision.models import resnet18, resnet34, resnet101, vgg16, vgg19, googlenet


def Network(name):
    if name == 'Resnet18':
        model = resnet18(pretrained=True)
    elif name == 'Resnet34':
        model = resnet34(pretrained=True)
    elif name == 'Resnet101':
        model = resnet101(pretrained=True) 
    elif name == 'VGG16':
        model = vgg16(pretrained=True) 
    elif name == 'VGG19':
        model = vgg19(pretrained=True)                     
    else:
        model = googlenet(pretrained=True)
    
    return model
