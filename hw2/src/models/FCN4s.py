import torch
from torch import nn
from torchvision import models
import numpy as np

class FCN4s(nn.Module):
    def __init__(self, num_class, dropout=0.1):
        super().__init__()
        self.__name__ = 'FCN4s'

        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        self.features2 = nn.Sequential(*features[:10])
        self.features3 = nn.Sequential(*features[10:17])
        self.features4 = nn.Sequential(*features[17:24])
        self.features5 = nn.Sequential(*features[24:])

        # ks=3, pad=1 keeps H, W
        self.fcn = nn.Sequential(      # conv 6-7
            nn.Conv2d(512, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        
        # for pool5
        self.score5 = nn.Conv2d(4096, num_class, kernel_size=1) # [-1, num_class, 7, 7]
        self.upsample_5 = nn.ConvTranspose2d(num_class, num_class, kernel_size=2, stride=2) # [-1, num_class, 14, 14]

        # for pool4
        self.score4 = nn.Conv2d(512, num_class, kernel_size=1) # [-1, num_class, 14, 14]
        self.upsample_4 = nn.ConvTranspose2d(num_class, num_class, kernel_size=2, stride=2) # [-1, num_class, 14, 14]

        # for pool3
        self.score3 = nn.Conv2d(256, num_class, kernel_size=1) # [-1, num_class, 28, 28]
        self.upsample_3 = nn.ConvTranspose2d(num_class, num_class, kernel_size=2, stride=2) # [-1, num_class, 14, 14]
        
        # for pool2
        self.score2 = nn.Conv2d(128, num_class, kernel_size=1) # [-1, num_class, 28, 28]
        self.upsample_2 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=4) # [-1, num_class, 14, 14]


        # Conv: ks=1 keep the H, W
        # DeConv: stride=kernel size equals to upsample rate for H, W        

    def forward(self, x):
        pool2 = self.features2(x)       # [-1, 128, 56, 56] 
        pool3 = self.features3(pool2)   # [-1, 256, 28, 28] (3 MaxPool: 224/8=28)
        pool4 = self.features4(pool3)   # [-1, 512, 14, 14]
        pool5 = self.features5(pool4)   # [-1, 4096, 7, 7]
        fcn32s_pre = self.fcn(pool5)    # [-1, 4096, 7, 7]
        
        fcn16s_in1 = self.upsample_5(self.score5(fcn32s_pre)) # [-1, num_class, 14, 14]
        fcn16s_in2 = self.score4(pool4) # [-1, num_class, 14, 14]
        fcn16s_pre =  fcn16s_in1 + fcn16s_in2 # [-1, num_class, 14, 14]
        # fcn16s = upsampling fcn16s_pre 16x

        fcn8s_in1 = self.upsample_4(fcn16s_pre) # [-1, num_class, 28, 28]
        fcn8s_in2 = self.score3(pool3) # [-1, num_class, 28, 28]
        fcn8s_pre = fcn8s_in1 + fcn8s_in2
        # fcn8s = upsampling fcn8s_pre 8x

        fcn4s_in1 = self.upsample_3(fcn8s_pre) # [-1, num_class, 56, 56]
        fcn4s_in2 = self.score2(pool2) # [-1, num_class, 56, 56]
        fcn4s_pre = fcn4s_in1 + fcn4s_in2 
        
        fcn4s = self.upsample_2(fcn4s_pre)  # [-1, num_class, 224, 224]

        return fcn4s


if __name__ == '__main__':
    
    from torchsummary import summary 
    device = 'cuda'
    
    model = FCN4s(7)

    model = model.to(device)
    #a = torch.rand(10, 3, 224, 224).to(device)
    #model(a)
    print(summary(model, (3, 224, 224), device=device))