import torch
import torch.nn as nn

class VAE_loss(nn.Module):
    def __init__(self, kld_weight=1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.kld_weight = kld_weight
    
    def forward(self, recon, x, mu, logvar):
        recons_loss = self.mse_loss(recon, x)
        L_kl = -0.5 * torch.sum(
            1 + logvar - mu ** 2 -logvar.exp(), 
            dim=1
        )
        kld_loss = torch.mean(L_kl, dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        
        return loss, recons_loss, kld_loss

if __name__ == "__main__":
    device = 'cuda'
    criterion = VAE_loss(1)
    a = torch.tensor([0.8, 0.1, 0.1])
    b = torch.tensor([0.85, 0.05, 0.1])
    mu = torch.tensor(0.1)
    logvar = torch.tensor(0.5)
    print(criterion(a, b, mu, logvar))