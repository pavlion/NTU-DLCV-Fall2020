import os
import torch
import torch.nn as nn


class Hallucinator(nn.Module):

    def __init__(self, m=10, hid_dim=1600):
        super().__init__()
        self.m = m
        self.hid_dim = hid_dim
        
        # Input = [1, way, hid_dim]
        # Output = [hid_dim]
        self.hall = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x = (1, way, hid_dim)
        seed = x.detach().repeat(self.m, 1, 1)
        noise = torch.rand_like(seed)
        x = torch.cat((seed, noise), dim=-1)
        x = self.hall(x) # (m, way, hid)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = Hallucinator().cuda()
    summary(model, (3, 64, 64))