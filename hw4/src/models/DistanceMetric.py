import os
import torch
import torch.nn as nn

class DistanceMetric(nn.Module):
    def __init__(self, distance_type='cosine', hid_dim=1):
        super().__init__()

        assert distance_type in ('cosine', 'euclid', 'parametric'), \
            print(f"Specified distance type: {distance_type} is wrong.")

        self.distance_type = distance_type
        self.hid_dim = hid_dim
        
        if distance_type == 'cosine':
            self.score = _cosine_metric

        elif distance_type == 'parametric':
            # input size = (len, hid_dim)
            self.score = ParamDist(hid_dim)

        else: # default to euclid
            self.score = _euclidean_metric 

    def forward(self, proto_q, proto_s):
        return self.score(proto_q, proto_s)

def _euclidean_metric(proto_q, proto_s):
    '''
    Compute euclidean distance between two tensors
    '''
    assert proto_s.size(1) == proto_q.size(1), \
        "Invalid input for computing euclidean distance"
    num_way = proto_s.size(0) 
    num_query = proto_q.size(0)
    
    proto_q = proto_q.unsqueeze(1).expand(num_query, num_way, -1)
    proto_s = proto_s.unsqueeze(0).expand(num_query, num_way, -1)

    return -torch.pow(proto_q - proto_s, 2).sum(2)

def _cosine_metric(proto_q, proto_s):
    '''
    Compute cosine distance between two tensors
    '''
    assert proto_s.size(1) == proto_q.size(1), \
        "Invalid input for computing euclidean distance"
    num_way = proto_s.size(0) 
    num_query = proto_q.size(0)
    
    proto_q = proto_q.unsqueeze(1).expand(num_query, num_way, -1)
    proto_s = proto_s.unsqueeze(0).expand(num_query, num_way, -1)

    return nn.CosineSimilarity(dim=2)(proto_q, proto_s) 


class ParamDist(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.param_dist = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, proto_q, proto_s):
        # proto_q: num_query x hid_dim
        # proto_s: num_way x hid_dim
        assert proto_s.size(1) == proto_q.size(1), \
            "Invalid input for computing parametric distance"
        num_way = proto_s.size(0) 
        num_query = proto_q.size(0)
        
        proto_q = proto_q.unsqueeze(1).repeat(1, num_way, 1)
        proto_s = proto_s.unsqueeze(0).repeat(num_query, 1, 1)
        # proto_q = proto_q.unsqueeze(1).expand(num_query, num_way, -1)
        # proto_s = proto_s.unsqueeze(0).expand(num_query, num_way, -1)

        x = torch.cat((proto_q, proto_s), dim=-1)
        x = self.param_dist(x)
        return x.squeeze(dim=-1)

if __name__ == '__main__':
    y = torch.tensor([3, 2, 1]).reshape(1, -1).float()
    x = torch.tensor([1, 2, 3]).reshape(1, -1).float()
    
    ans = {'cosine': 10, 'euclid': 8, 'parametric': 0}
    for type in ('cosine', 'euclid', 'parametric'):
        print(type)
        print(f"{type}:", DistanceMetric(type, hid_dim=3)(x, y), ", Ans:", ans[type])