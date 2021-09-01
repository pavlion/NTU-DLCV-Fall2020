import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, fc_dim=64, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        # [bs, 3, 84, 84]
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

        # [bs, 1600]
        self.score = nn.Sequential(
            nn.Linear(1600, fc_dim*2),
            nn.Linear(fc_dim*2, fc_dim)
        ) # [bs, fc_dim]

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.score(x)
        return x

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    return block
