import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
#import wandb
from .logger import wandb
#from utils.tensorboard_logger import Logger

from .GAN import Generator, Discriminator, init_weights

torch.manual_seed(422)
def mean(x): return sum(x)/len(x)

wandb.init(project="dlcv_3-2")


class DCGAN_Config:
    def __init__(
        self,
        batch_size=32,
        latent_size=100,
        epochs=1000,
        learning_rate=0.002
    ):
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def get_cfg_dict(self):
        cfg = {
            'batch_size': self.batch_size,
            'latent_size': self.latent_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }

        return cfg


class DCGAN():
    def __init__(self, cfg=DCGAN_Config()):
        print("DCGAN model initalization.")
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("***Torch seed:", torch.initial_seed())

        # Model init
        self.G = Generator(latent_size=cfg.latent_size).to(self.device)
        self.D = Discriminator().to(self.device)
        self.G.apply(init_weights)
        self.D.apply(init_weights)
        self.latent_size = cfg.latent_size
        self.real_label = 1.
        self.fake_label = 0.

        self.fixed_noise = torch.randn(
            (cfg.batch_size, cfg.latent_size, 1, 1)
        ).to(self.device)

        # Training settings
        # Using lower learning rate than suggested by (ADAM authors)
        # lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.criterion = nn.BCEWithLogitsLoss()
        self.d_optimizer = optim.Adam(
            self.D.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(
            self.G.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size

        # Set the logger
        #self.logger = Logger('./logs')
        self.number_of_images = 10
        self.steps = 0
        self.max_loss_G = 0.0
        self.loss_history = {'G': [], 'D': []}
        self.img_list = []

    def train(self, train_loader, dest_dir):
        ''' Train DCGAN with `train_loader` '''
        self.G.train(True)
        self.D.train(True)
        for epoch in range(self.epochs):
            epoch_loss = {'D_loss': [], 'G_loss': [], 'D_x': [], 'D_G_z': []}
            for i, batch in enumerate(train_loader):
                images = batch['img'].to(self.device)
                batch_size = images.size(0)
                real_labels = torch.full(
                    (batch_size, ), self.real_label, dtype=torch.float, device=self.device
                )
                fake_labels = torch.full(
                    (batch_size, ), self.fake_label, dtype=torch.float, device=self.device
                )
                z = torch.randn((batch_size, self.latent_size, 1, 1), device=self.device)  # noqa

                D_loss, D_x, D_G_z1 = self.train_discriminator(
                    images, z, real_labels, fake_labels)
                G_loss, D_G_z2 = self.train_generator(z, real_labels)

                # Log info:
                self.steps += 1
                epoch_loss['D_loss'].append(D_loss)
                epoch_loss['G_loss'].append(G_loss)
                epoch_loss['D_x'].append(D_x)
                epoch_loss['D_G_z'].append(D_G_z2)

                print('Epoch {:3d} ({:3d}/{:3d}) | Loss_D: {:.4f} | Loss_G: {:.4f} | D(x): {:.4f} | D(G(z)): {:.4f}/{:.4f}'
                      .format(epoch, i, len(train_loader),
                              D_loss, G_loss, D_x, D_G_z1, D_G_z2) + " "*10, end='\r')

                wandb.log({
                    'D_loss': D_loss,
                    'G_loss': G_loss,
                    'D_x': D_x,
                    'D_G_z': D_G_z2
                })


                # Check how the generator is doing by saving G's output on fixed_noise
                if self.steps % 500 == 0 or i == len(train_loader)-1:
                    
                    with torch.no_grad():
                        # z = torch.randn((self.cfg.batch_size, self.latent_size, 1, 1), device=self.device)
                        # fake_random = self.G(z).detach().cpu()
                        fake_fixed = self.G(self.fixed_noise).detach().cpu()

                    vutils.save_image(
                        fake_fixed, 
                        fp=os.path.join(dest_dir, "gen_img", f"fixed_epoch{epoch}_steps{self.steps}.png"),
                        normalize=True,
                        range=(-1, 1)
                    )
                    # vutils.save_image(
                    #     fake_random, 
                    #     fp=os.path.join(dest_dir, "gen_img", f"random_generated.png"),
                    #     normalize=True,
                    #     range=(-1, 1)
                    # )

                    wandb.log({
                        'gen_image': wandb.Image(
                            vutils.make_grid(
                            fake_fixed,
                            normalize=True,
                            range=(-1, 1)
                        )
                    )})

            self.loss_history['D'] += epoch_loss['D_loss']
            self.loss_history['G'] += epoch_loss['G_loss']
            print('Epoch {:3d} | Loss_D: {:.4f} | Loss_G: {:.4f} | D(x): {:.4f} | D(G(z)): {:.4f}'
                  .format(epoch, mean(epoch_loss['D_loss']), mean(epoch_loss['G_loss']),
                          mean(epoch_loss['D_x']), mean(epoch_loss['D_G_z']))+" "*50, end='\n')
            
            if mean(epoch_loss['G_loss']) > 9.0:
                torch.save(self.G.state_dict(),os.path.join(dest_dir, f"G_epoch{epoch}.pth"))
                print("***Model saved.***")


    def train_discriminator(self, images, z, real_labels, fake_labels):
        '''
        Train discriminator for one iteration. 

        Args: 
            images: target image = (B, C, H, W)
            z: randomly-generated latent vector = (B, Z, 1, 1)
            real_labels: real labels (B, )
            fake_labels: fake labels (B, )
        Return:
            D_loss (float): discriminator loss
            D_x (float): discriminator score on real batch
            D_G_z1 (float): discriminator score on fake batch
        '''
        real_outputs = self.D(images).view(-1) 
        fake_images = self.G(z)
        fake_outputs = self.D(fake_images.detach()).view(-1)  # detach from graph

        # Loss
        d_loss_real = self.criterion(real_outputs, real_labels)
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # Back Prop
        self.D.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Return discriminator info
        D_loss = d_loss.item()
        D_x = torch.nn.functional.sigmoid(real_outputs).mean().item()
        D_G_z1 = torch.nn.functional.sigmoid(fake_outputs).mean().item()
        #print(d_loss_real.item(), d_loss_fake.item())
        return D_loss, D_x, D_G_z1

    def train_generator(self, z, real_labels):
        '''
        Train generator for one iteration. 

        Args: 
            z: latent vector = (B, Z, 1, 1)
            real_labels: real labels (B, )
        Return:
            G_loss (float): generator loss
            D_G_z2 (float): discriminator score on fake batch
        '''

        fake_images = self.G(z)
        outputs = self.D(fake_images).view(-1)

        g_loss = self.criterion(outputs, real_labels)

        self.G.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # Return generator info
        G_loss = g_loss.item()
        D_G_z2 = torch.nn.functional.sigmoid(outputs).mean().item()

        return G_loss, D_G_z2
