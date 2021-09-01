import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as tv_models 

class DeepLabv3_ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.__name__ = "DeepLabv3_ResNet101"
        self.backbone = tv_models.deeplabv3_resnet101(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)['out']
        return x