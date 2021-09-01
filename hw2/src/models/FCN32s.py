import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FCN32s(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.__name__ = 'FCN32s'

        self.backbone = models.vgg16(pretrained=True).features  # [32, 512, 7, 7]
        self.conv_block = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.classifier = nn.Sequential(            
            nn.Conv2d(4096, num_class, kernel_size=1),
            nn.ReLU(inplace=True)    
        )
        self.upsample = nn.ConvTranspose2d(
            num_class, num_class, kernel_size=32, stride=32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_block(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x


class SegNet32(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.__name__ = 'VGG16_FCN32s'
        self.backbone = models.vgg16(
            pretrained=True).features  # [32, 512, 7, 7]
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_class, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_block(x)
        return x


if __name__ == '__main__':

    #from torchsummary import summary
    #model = SegmentationModel(7).cuda()
    #print(summary(model, (3, 224, 224)))
    # print(model.__name__)
    #print(model)

    from model_summary import summary 
    model = FCN32s(7)
    model_summary = summary(model, (3, 224, 224), device='cpu')
    with open("p2_model.txt", "w") as f:
        f.write(model_summary)
        f.write("\n\n\n")
        print(model, file=f)
