import os
import torch
import torch.nn as nn
from torch.autograd import Function

class DANN(nn.Module):
    def __init__(self, num_filters=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # input: [3, 28, 28]
            nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=5, stride=1, padding=1), 
            nn.BatchNorm2d(num_filters), # [64, 26, 26]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25),

            # maps: [64, 13, 13]
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=5, stride=1, padding=1), # maps: [64, 8, 8]
            nn.BatchNorm2d(num_filters*2),   # [128, 11, 11] 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.3),        
            # maps: [128, 5, 5]
            
            Flatten(),
            # vec: [3200] (=128*5*5)
        )
        feature_size = num_filters*2*5*5
        self.label_classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.feature_size = feature_size
        #self._init_weight()

    def forward(self, x, alpha=0.1, reverse=True):
        features = self.feature_extractor(x)

        # predict class
        label = self.label_classifier(features)

        # predict domain
        if reverse: 
            reversed_feature = ReverseLayerF.apply(features, alpha)
            domain = self.domain_classifier(reversed_feature)
        else:
            domain = self.domain_classifier(features)

        return label, domain

    def _init_weight(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

class DANN_ResNet(nn.Module):
    def __init__(self, n_features=64, latent_dim=512, p=0.1):
        super().__init__()
        import torchvision.models as models 

        self.n_features = n_features

        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *(list(backbone.children())[:-3]),
            Flatten(),
            nn.Linear(1024, latent_dim),
            nn.ReLU(True),
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 10)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1)
        )

    def forward(self, X, alpha=0.5):
        features = self.feature_extractor(X)

        # label classification
        label_output = self.label_predictor(features)

        # domain classification
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return label_output, domain_output

    def extract_feature(self, X):
        features = self.feature_extractor(X)

        return features

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        
        return output, None

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
if __name__ == '__main__':
    from torchsummary import summary
    model = DANN().cuda()
    #model = DANN_ResNet().cuda()
    #a = torch.rand((32, 3, 28, 28)).cuda()
    #model(a)
    summary(model, (3, 28, 28), batch_size=16)