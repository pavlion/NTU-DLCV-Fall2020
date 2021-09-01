import os
import torch
import torch.nn as nn

from .DistanceMetric import DistanceMetric
from .FeatureExtractor import FeatureExtractor

class DTN(nn.Module):
    '''
    Args:
        hid_dim: dimension for hidden vectors
        m: # of images augmented in each batch (default=0, not hall)
        distance_type: what kind of distance metric is used
    '''
    def __init__(self, hid_dim=64, hall=True, distance_type='cosine'):
        super().__init__()
        self.hid_dim = hid_dim
        self.hall = hall
        self.distance_type = distance_type

        self.distance = DistanceMetric(distance_type, hid_dim)
        self.encoder = FeatureExtractor(fc_dim=hid_dim) # (bs, hid_dim)
        self.hallucinator = DTNHallucinator(hid_dim)

        self.apply(init_weights)

        # for m in self.children():
        #     init_weights(m)

    def forward(self, shot, query, ref1, ref2, num_way):
        # Input:  shot = (shot*way, 3, 84, 84)
        # Input:  query = (query, )
        # Input:  ref1, ref2 = (H, 3, 84, 84)
        # Output: (query, way)

        num_shot = shot.size(0) // num_way 
        num_query = query.size(0)
        proto_s = self.encoder(shot) # (shot*num_way, hid_dim)
        proto_q = self.encoder(query) # (query, hid_dim)
        proto_s = proto_s.reshape(num_shot, num_way, -1)

              
        # Randomly select a data as seed
        # Hallucinate it and concat with original shot 
        if self.hall:
            ind = torch.randperm(num_shot)
            support_feat = proto_s[ind].view(1, num_way, -1).detach() # (way, hid_dim)
            ref_feat1 = self.encoder(ref1) # (H, hid_dim)
            ref_feat2 = self.encoder(ref2)        
            imaginary_data = self.hallucinator(support_feat,
                                ref_feat1, ref_feat2) # (H, way, hid_dim)

            proto_s = torch.cat((proto_s, imaginary_data), dim=0)
        
        # Average the prototypes within shot => (num_way, hid_dim)
        proto_s = proto_s.mean(dim=0)
        logits = self.distance(proto_q, proto_s)

        return logits


class DTNHallucinator(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.diversify = AddDiversity(hid_dim)
        self.generator = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, support_feat, ref_feat1, ref_feat2):
        # support_feat: (1, way, hid_dim)
        # ref_feat1, ref_feat2: (H, hid_dim)
        # This will create H new features
        num_way = support_feat.size(1)
        ref_feat1 = ref_feat1.unsqueeze(1).expand(-1, num_way, -1)
        ref_feat2 = ref_feat2.unsqueeze(1).expand(-1, num_way, -1)

        diversified = self.diversify(support_feat, ref_feat1, ref_feat2)
        rebuild = self.generator(diversified)
        #rebuild = self.l2_norm(rebuild)

        return rebuild # (H, way, hid_dim)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        norm = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(norm)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output


class AddDiversity(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(hid_dim, hid_dim*2),
            nn.LeakyReLU(0.2, inplace=True)
        )        
        self.dropout = nn.Dropout(0.5)

    def forward(self, A, B1, B2):
        A, B1, B2 = self.encode(A), self.encode(B1), self.encode(B2)
        out = A + (B1 - B2)
        out = self.dropout(out)
        return out


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    from torchsummary import summary
        # Input:  shots = (shot*way, 3, 84, 84)
        # Input:  query = (query, 3, 84, 84)
        # Input:  ref1, ref2 = (H, 3, 84, 84)
        # Output: (query, way)
    shot = 1
    query = 15
    way = 5
    H = 100
    hid_dim = 64
    shot = torch.rand((shot*way, 3, 84, 84)).cuda()
    query = torch.rand((query, 3, 84, 84)).cuda()
    ref_feat1 = torch.rand((H, 3, 84, 84)).cuda()
    ref_feat2 = torch.rand((H, 3, 84, 84)).cuda()


    model = DTN(64).cuda()
    logits = model(shot, query, ref_feat1, ref_feat2, way)
    print(logits.shape)
    #print(model)