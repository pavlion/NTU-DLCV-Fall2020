import os
import torch
import torch.nn as nn

from .DistanceMetric import DistanceMetric
from .FeatureExtractor import FeatureExtractor

class IFINet(nn.Module):
    '''
    Imagine from imagine network.
    Args:
        hid_dim: dimension for hidden vectors
        m: # of images augmented in each batch (default=0, not hall)
        distance_type: what kind of distance metric is used
    '''
    def __init__(self, hid_dim=64, m=0, distance_type='euclid'):
        super().__init__()
        self.hid_dim = hid_dim
        self.distance_type = distance_type

        self.distance = DistanceMetric(distance_type, hid_dim)
        self.encoder = FeatureExtractor(fc_dim=hid_dim) # (bs, hid_dim)
        self.hallucinator = IFIHallucinator(m, hid_dim)

        self.apply(init_weights)

        # Input:  shots = (shot*way, 3, 84, 84)
        # Input:  query = (query, )
        # Output: (query, way)

    def forward(self, shots, query, num_way):
        num_shot = shots.size(0) // num_way 
        num_query = query.size(0)
        proto_s = self.encoder(shots) # (shot*num_way, hid_dim)
        proto_q = self.encoder(query) # (query, hid_dim)

        proto_s = proto_s.reshape(num_shot, num_way, -1)

        if self.m != 0:           
            # Randomly select a data as seed
            # Hallucinate it and concat with original shot 
            idx = torch.randperm(num_shot)
            imaginary_data = self.hallucinator(proto_s[idx]) # (m, way, hid)
            ifi_data = self.hallucinator(imaginary_data[idx]) # (m, way, hid)
            proto_s = torch.cat((proto_s, imaginary_data, ifi_data), dim=0)
        
        # Average the prototypes within shot => (num_way, hid_dim)
        proto_s = proto_s.mean(dim=0)
        logits = self.distance(proto_q, proto_s)
        
        return logits

class IFIHallucinator(nn.Module):

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



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    from torchsummary import summary
    model = PrototypicalNet().cuda()
    summary(model, (3, 84, 84))