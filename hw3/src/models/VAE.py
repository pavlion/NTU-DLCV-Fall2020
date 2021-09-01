import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.__name__ = 'VAE'
        self.hidden_size = hidden_size

        self.encode_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
 
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
 
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.ConvTranspose2d(32, 3, 4, 2, padding=1,bias=False)
        )

        self.fc_mu = nn.Linear(4096, hidden_size)
        self.fc_logvar = nn.Linear(4096, hidden_size)
        self.fc_decode = nn.Linear(hidden_size, 4096)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar

    def encode(self, x):
        """
        Encodes the input with encoder block and returns the latent codes
        Args:
            x (Tensor): Input tensor to encoder [N x C x H x W]
        Return: 
            (Tensor) List of latent codes
        """
        x = self.encode_block(x).view(-1, 4096)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        """
        Maps the given latent codes onto the image space
        Args:
            z (Tensor): [B x D]
        Return: 
            (Tensor) [B x C x H x W]
        """
        z = self.fc_decode(z).view(-1, 256, 4, 4)
        z = self.decode_block(z)        
        return  torch.tanh(z)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization to sample from N(mu, var) from N(0,1).
        Args:
            mu (Tensor): Mean of the latent Gaussian [B x D]
            logvar (Tensor): Standard deviation of the latent Gaussian [B x D]
        Return: 
            (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # sample from N(0, 1)
        return eps * std + mu
    
    def sample(self, num_samples, device='cuda'):
        """
        Samples from the latent space and return the corresponding
        image space map.
        Args:
            num_samples: (Int) Number of samples
            current_device: (Int) Device to run the model
        Return: (Tensor)
        """
        z = torch.randn(num_samples, self.hidden_size).to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        Args
            x (Tensor): [B x C x H x W]
        Return (Tensor): [B x C x H x W]
        """
        recon_img, _, _ = self.forward(x)
        return recon_img
    
    def get_latent_vec(self, x):
        """
        Given an input image x, returns its corresponding latent vector
        Args
            x (Tensor): [B x C x H x W]
        Return (Tensor): [B x C x H x W]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

  

if __name__ == '__main__':

    from torchsummary import summary 
    model = VAE(7).cuda()
    summary(model, (3, 64, 64))
    