import os
import torch
import torch.nn as nn


class Generator(torch.nn.Module):
    def __init__(self, latent_size=100, out_channels=3):
        super().__init__()
        self.latent_size = latent_size
        # Filters: 1024, 512, 256
        # Input = [-1, latent_size, 1, 1]
        # Output = [out_channels]
        self.main_module = nn.Sequential(
            # latent vector z = [latent_size]
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # Maps = [1024, 4, 4]
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # Maps = [512, 8, 8]
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # Maps = [256, 16, 16]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            
            # Maps = [128, 32, 32]
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            
            # Maps = [C_out, 64, 64] =  generated image
            )

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)



class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3, LR_rate=0.2):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input = [C, 64, 64]
        # Output = [1]
        self.main_module = nn.Sequential( 
            # Image = [C, 64, 64] 
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256), # not appeared in tutorial
            nn.LeakyReLU(LR_rate, inplace=True),

            # Maps = [256, 32, 32]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(LR_rate, inplace=True),

            # Maps = [512, 16, 16]
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(LR_rate, inplace=True),

            # Maps = [1024, 8, 8]
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=2048),
            nn.LeakyReLU(LR_rate, inplace=True)
            # Maps = [2048, 4, 4]
        )

        self.output = nn.Sequential(
            # The output of D is no longer a probability, 
            # we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=4, stride=1, padding=0),
            # nn.Sigmoid()
        ) # [2048, 1, 1]

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten the vector
        x = self.main_module(x)
        return x.view(-1, 2048*4*4)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    from torchsummary import summary
    model = CNNModel().cuda()
    summary(model, (3, 64, 64))