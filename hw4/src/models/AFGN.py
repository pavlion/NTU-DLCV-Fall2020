import os
import torch
import torch.nn as nn

from .Hallucinator import Hallucinator
from .DistanceMetric import DistanceMetric
from .FeatureExtractor import FeatureExtractor

class AFHN(nn.Module):
    '''
    Adversarial Feature Hallucination Network.

    Args:
        hid_dim: dimension for hidden vectors
        m: # of images augmented in each batch (default=0, not hall)
        distance_type: what kind of distance metric is used
    
    Return in Forward:
        logits: logits for classifier
        seed: s, real features (selected seed)
        imaginary_data: s_tilda, generated features
        noise: z, sampled noise
    '''
    def __init__(self, hallucinator, discriminator, hid_dim=64, m=0):
        super().__init__()
        self.hid_dim = hid_dim
        self.m = m  
        self.distance_type = 'cosine'

        self.distance = DistanceMetric(self.distance_type, hid_dim)
        self.encoder = FeatureExtractor(fc_dim=hid_dim) # (bs, hid_dim)
        self.hallucinator = hallucinator
        self.discriminator = discriminator

        self.apply(init_weights)
        # for m in self.children():
        #     init_weights(m)

        # Input:  shots = (shot*way, 3, 84, 84)
        # Input:  query = (query, )
        # Output: (query, way)

    def forward(self, shots, query, num_way):
        num_shot = shots.size(0) // num_way 
        num_query = query.size(0)
        
        # proto_s = (shot, num_way, hid_dim)
        # proto_q = (query, hid_dim)
        proto_s = self.encoder(shots).reshape(num_shot, num_way, -1) 
        proto_q = self.encoder(query)


        # Randomly select a data as seed
        # Hallucinate it and concat with original shot 
        ind = torch.randperm(num_shot)
        seed = proto_s[ind].detach().expand(self.m, 1, 1) # (m, way, hid)
        noise = torch.rand_like(seed) # (m, num_way, hid_dim)
        
        imaginary_data = self.hallucinator(   
            torch.cat((seed, noise), dim=-1)) # (m, way, hid)

        proto_s = torch.cat((proto_s, imaginary_data), dim=0)
    
        # Average the prototypes within shot => (num_way, hid_dim)
        proto_s = proto_s.mean(dim=0)
        logits = self.distance(proto_q, proto_s)
        
        return logits, seed, imaginary_data, noise


class Generator(nn.Module):
    def __init__(self, in_features=1600, LR_rate=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features*2, in_features),
            nn.LeakyReLU(LR_rate),
            nn.Linear(in_features, in_features),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_features=1600, hid_dim=1024, LR_rate=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features*2, hid_dim),
            nn.LeakyReLU(LR_rate),
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, s, z):
        # s, z: (m, way, hid_dim)
        x = torch.cat((s, z), dim=-1)
        x = self.mlp(x)
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