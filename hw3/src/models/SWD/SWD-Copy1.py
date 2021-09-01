import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Function

class SWD:

    class Generator(nn.Module):
        def __init__(self, num_filters=64):
            super().__init__()
            # input = [3, 28, 28]
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=5, stride=1, padding=2), # [64, 24, 24]
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                
                nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2), # [64, 24, 24]
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                
                nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2), # [64, 24, 24]
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                
                # map = [64, 12, 12]
                nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=5, stride=1, padding=2), 
                nn.BatchNorm2d(num_filters*2), # map = [128, 8, 8]
                nn.ReLU(inplace=True)
                # map = [128, 8, 8]
            )
            
            self.fc = nn.Sequential(
                
            )

        def forward(self, x):
            x = self.conv_blocks(x).view(-1, 8192)
            x = self.fc(x)
            return x # [bs, feature_size]

    class Classifier(nn.Module):
        def __init__(self, in_feature=3072, Lambda=1):
            super().__init__()
            self.main_module = nn.Sequential(
                nn.Linear(8192, 3072),
                nn.BatchNorm1d(3072), 
                nn.ReLU(inplace=True),               
                nn.Dropout(),
                
                nn.Linear(3072, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 10),
            )
            self.Lambda = Lambda

        def forward(self, x, reverse=False):
            if reverse:
                x = SWD.ReverseLayerF.apply(x, self.Lambda)

            x = self.main_module(x)
            return x

    class ReverseLayerF(Function):

        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha

            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.alpha
            
            return output

    class USPS:

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()                
                self.block1 = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=5, stride=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1))
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(32, 48, kernel_size=5, stride=1),
                    nn.BatchNorm2d(48),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1))
                )

            def forward(self, x):
                B, _, H, W = x.size()
                x = torch.mean(x,1).view(B, 1, H, W)
                x = self.block1(x)
                x = self.block2(x)
                #print(x.size())
                x = x.view(B, 48*4*4)
                return x


        class Classifier(nn.Module):
            def __init__(self, prob=0.0):
                super().__init__()
                feature_size = 48*4*4
                self.dropout1 = nn.Dropout(prob)
                self.block = nn.Sequential(
                    nn.Linear(feature_size, 100),
                    nn.BatchNorm1d(100),
                    nn.ReLU(True),
                    nn.Linear(100, 100),
                    nn.BatchNorm1d(100),
                    nn.Linear(100, 10)
                )
                
                self.prob = prob

            def forward(self, x, alpha=1,reverse=False):
                if reverse:
                    x = SWD.ReverseLayerF.apply(x, alpha)
                x = self.block(x)
                return x


class discrepancy_slice_wasserstein(nn.Module):
    def __init__(self, M=128):
        self.M = M
        ''' M = number of radial projections'''
    
    def __call__(self, p1, p2):
        p1, p2 = F.sigmoid(p1), F.sigmoid(p2)

        s = p1.shape
        if s[1]>1:
            proj = torch.randn(s[1], self.M).to(p1.device)
            proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
            p1 = torch.matmul(p1, proj)
            p2 = torch.matmul(p2, proj)
            #print("proj p1 shape:", p1.shape)
        p1 = torch.topk(p1, s[0], dim=0)[0]
        p2 = torch.topk(p2, s[0], dim=0)[0]
        #print("topk p1 shape:", p1.shape)
        dist = p1-p2
        wdist = torch.mean(torch.mul(dist, dist))
        #print("wdist shape:", p1.shape)
        
        return wdist
    

def discrepancy_mcd(out1, out2):
    out1, out2 = F.softmax(out1), F.softmax(out2)
    return torch.mean(torch.abs(out1 - out2))
      
if __name__ == "__main__":
    from torchsummary import summary
    
    class _test(nn.Module):
        def __init__(self, in_feature=2048):
            super().__init__()           
            self.G = SWD.USPS.Generator()
            self.C1 = SWD.USPS.Classifier()

        def forward(self, x):
            x = self.G(x)
            x = self.C1(x)
            return x

    model = _test().cuda()
    summary(model, (3, 28, 28), batch_size=16)