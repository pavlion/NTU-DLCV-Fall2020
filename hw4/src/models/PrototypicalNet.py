import os
import torch
import torch.nn as nn

from .Hallucinator import Hallucinator
from .DistanceMetric import DistanceMetric
from .FeatureExtractor import FeatureExtractor

class PrototypicalNet(nn.Module):
    '''
    Args:
        hid_dim: dimension for hidden vectors
        m: # of images augmented in each batch (default=0, not hall)
        distance_type: what kind of distance metric is used
    '''
    def __init__(self, hid_dim=64, m=0, ifi=False, distance_type='euclid'):
        super().__init__()
        self.hid_dim = hid_dim
        self.m = m  
        self.ifi = ifi
        self.distance_type = distance_type

        self.distance = DistanceMetric(distance_type, hid_dim)
        self.encoder = FeatureExtractor(fc_dim=hid_dim) # (bs, hid_dim)
        
        if self.m != 0:   
            self.hallucinator = Hallucinator(m, hid_dim)

        self.apply(init_weights)

        # for m in self.children():
        #     init_weights(m)

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
            ind = torch.randperm(num_shot)
            seed = proto_s[ind] 
            imaginary_data = self.hallucinator(seed) # (m, way, hid)
            proto_s = torch.cat((proto_s, imaginary_data), dim=0)
            
            if self.ifi:
                imagine_imaginary_data = self.hallucinator(imaginary_data[ind])
                proto_s = torch.cat((proto_s, imagine_imaginary_data), dim=0)
        
        # Average the prototypes within shot => (num_way, hid_dim)
        proto_s = proto_s.mean(dim=0)
        logits = self.distance(proto_q, proto_s) # (query, num_way)
        

        # logits = []
        # for proto in proto_s: # proto of each way: (hid_dim)
        #     proto = torch.cat([proto for _ in range(num_query)]).view(num_query, -1)
        #     distance = self.distance(proto, proto_q).reshape(-1, 1) # (query, )
        #     logits.append(distance) 

        # logits = torch.cat(logits, dim=1)  # (query, way)

        return logits

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ProtoNet_tSNE(nn.Module):
    '''
    Args:
        hid_dim: dimension for hidden vectors
        m: # of images augmented in each batch (default=0, not hall)
        distance_type: what kind of distance metric is used
    '''
    def __init__(self, hid_dim=64, m=0, ifi=False, distance_type='euclid'):
        super().__init__()
        self.hid_dim = hid_dim
        self.m = m  
        self.ifi = ifi
        self.distance_type = distance_type

        self.distance = DistanceMetric(distance_type, hid_dim)
        self.encoder = FeatureExtractor(fc_dim=hid_dim) # (bs, hid_dim)
        
        if self.m != 0:   
            self.hallucinator = Hallucinator(m, hid_dim)

        self.apply(init_weights)

        # for m in self.children():
        #     init_weights(m)

        # Input:  shots = (shot*way, 3, 84, 84)
        # Input:  query = (query, )
        # Output: (query, way)

    def forward(self, shots, query, num_way):
        num_shot = shots.size(0) // num_way 
        num_query = query.size(0)
        proto_s = self.encoder(shots) # (shot*num_way, hid_dim)
        proto_q = self.encoder(query) # (query, hid_dim)

        proto_s = proto_s.reshape(num_shot, num_way, -1)

        imaginary_data, imagine_imaginary_data = torch.FloatTensor([]), torch.FloatTensor([])
        if self.m != 0:           
            # Randomly select a data as seed
            # Hallucinate it and concat with original shot 
            ind = torch.randperm(num_shot)
            seed = proto_s[ind] 
            imaginary_data = self.hallucinator(seed) # (m, way, hid)
            
            if self.ifi:
                imagine_imaginary_data = self.hallucinator(imaginary_data[ind])
        
        
        
        return proto_s, imaginary_data, imagine_imaginary_data


if __name__ == '__main__':
    from torchsummary import summary
    shots = torch.rand((10, 3, 84, 84)).cuda()
    query = torch.rand((15, 3, 84, 84)).cuda()
    num_way = 5
    model = PrototypicalNet(distance_type='parametric').cuda()
    logits = model(shots, query, num_way)
    # print(logits.shape)
    print(model)