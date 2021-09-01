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
        def __init__(self, in_feature=3072, alpha=1):
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
            self.alpha = alpha

        def forward(self, x, reverse=False):
            if reverse:
                x = SWD.ReverseLayerF.apply(x, self.alpha)

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
                self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
                self.bn2 = nn.BatchNorm2d(48)

            def forward(self, x):
                x = torch.mean(x,1).view(x.size()[0],1,x.size()[2],x.size()[3])
                x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
                x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
                #print(x.size())
                x = x.view(x.size(0), 48*4*4)
                return x

        class Classifier(nn.Module):
            def __init__(self, prob=0.0):
                super().__init__()
                self.fc1 = nn.Linear(48*4*4, 100)
                self.bn1_fc = nn.BatchNorm1d(100)
                self.fc2 = nn.Linear(100, 100)
                self.bn2_fc = nn.BatchNorm1d(100)
                self.fc3 = nn.Linear(100, 10)
                self.bn_fc3 = nn.BatchNorm1d(10)
                self.prob = prob
                self.lambd = 1

            def forward(self, x, reverse=False):
                if reverse:
                    x = SWD.ReverseLayerF.apply(x, self.lambd)
                x = F.dropout(x, training=self.training, p=self.prob)
                x = F.relu(self.bn1_fc(self.fc1(x)))
                x = F.dropout(x, training=self.training, p=self.prob)
                x = F.relu(self.bn2_fc(self.fc2(x)))
                x = F.dropout(x, training=self.training, p=self.prob)
                x = self.fc3(x)
                return x

class SWD_test(nn.Module):
    def __init__(self, model_type='mnistm', in_feature=2048):
        super().__init__()      

        if model_type == 'usps':     
            self.G = SWD.USPS.Generator()
            self.C1 = SWD.USPS.Classifier()
            self.C2 = SWD.USPS.Classifier()
        else:
            self.G = SWD.Generator()
            self.C1 = SWD.Classifier()
            self.C2 = SWD.Classifier()

    def forward(self, x):
        x = self.G(x)
        x1 = self.C1(x)
        x2 = self.C1(x)
        return x1 + x2


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