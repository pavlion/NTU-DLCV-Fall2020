import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

class ResNext101(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.__name__ = "resnext101"
        self.backbone = models.resnext101_32x8d(pretrained=True)
        self.backbone.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.__name__ = "resnet18"
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResNet18_tSNE(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.resnet = ResNet18(out_dim=50) 

        modules = list(self.resnet.backbone.children())
        self.features = nn.Sequential(*modules[:-2])
        self.classifier = nn.Sequential(*modules[-2:])

    def forward(self, x):
        pred = self.features(x)
        # not producing the prediction by using self.classifier
        return pred


class ResNet34(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.__name__ = "resnet34"
        self.backbone = models.resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return x

class VGG16(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.__name__ = "VGG16"
        self.backbone = models.vgg16(pretrained=True)
        self.backbone.classifier[-1] = nn.Linear(4096, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == '__main__':
    
    from model_summary import summary 
    model = ResNet18(50)
    #model = VGG16(50)
    model_summary = summary(model, (3, 224, 224), device='cpu')
    with open("p1_model.txt", "w") as f:
        f.write(model_summary)
        f.write("\n\n\n")
        print(model, file=f)
    #print(model.__name__)
    
    
    
    
    

