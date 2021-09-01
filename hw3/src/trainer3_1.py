import os
import time
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.utils as vutils
import numpy as np

from trainer import Trainer
from utils import TrainingLogger, Accuracy
from loss import VAE_loss

mean = lambda x: sum(x)/len(x) 

class P1Trainer(Trainer):    

    def __init__(self,
                 max_epochs=10,
                 device=None,
                 learning_rate=1e-3,
                 kld_weight=0.1,
                 optimizer_opt='Adam',
                 model=None,
                 train_loader=None,
                 val_loader=None):
                 
        super().__init__(max_epochs=max_epochs,
                 device=device,
                 learning_rate=learning_rate,
                 optimizer_opt=optimizer_opt,
                 model=model,
                 train_loader=train_loader,
                 val_loader=val_loader)

        self.criterion = VAE_loss(kld_weight=kld_weight)
        self.kld_weight = kld_weight
        
        print("Torch seed:", torch.initial_seed())   
    
    def _run_epoch(self, dataloader, is_training):

        description = 'Train' if is_training else 'Val'
        
        training_history = {'MSE': [], 'KLD': []}
        loss = 0
        for i, batch in enumerate(dataloader):

            if is_training:
                self.model.train(True)

                # Forward propagation
                img = batch['img'].to(self.device)
                recon, mu, logvar = self.model(img) 

                # Backward propagation
                self.optimizer.zero_grad()
                batch_loss, recons_loss, kld = self.criterion(recon, img, mu, logvar)
                batch_loss.backward()
                self.optimizer.step()

            else:
                self.model.train(False)
                with torch.no_grad():
                    img = batch['img'].to(self.device)
                    recon, mu, logvar = self.model(img)  
                    batch_loss, recons_loss, kld = self.criterion(recon, img, mu, logvar)

                    
            loss += batch_loss.item()
            training_history['MSE'].append(float(recons_loss))
            training_history['KLD'].append(float(kld))
            
            print("{:5s} | Steps:{:4d}/{:-4d}  |  loss={:.6f} | MSE={:.4f} | KLD={:.4f}".format(description, 
                i, len(dataloader), loss/(i+1),                                                                                mean(training_history['MSE']), mean(training_history['KLD'])), end='\r')
            
            
            
            
        loss /= len(dataloader)
        epoch_log = {
            'loss': float(loss),
            'MSE_loss': training_history['MSE'], 
            'KLD': training_history['KLD']
        }
        self.saved_model_name = f"best_loss_MSE{epoch_log['MSE_loss']}_KLD{epoch_log['KLD']}.pth"
        return epoch_log

    def epoch_logger(self, log_train, log_valid):
        '''
        Message to be printed after each epoch. (Override parent method.)
        '''
        mean = lambda x: sum(x)/len(x) 
        msg = "train loss={:.6f} | val loss={:.6f} | MSE: {:.4f} | KLD: {:.4f}".format(
            log_train['loss'], 
            log_valid['loss'],
            mean(log_valid['MSE_loss']),
            mean(log_valid['KLD'])
            ) + " "*50
        
        return msg
        
    def predict(self, test_loader=None, num_samples=0):
        self.model.eval()
        generated_imgs = [] 
        with torch.no_grad():
            if test_loader is None:
                samples = self.model.sample(
                    num_samples=num_samples, 
                    device=self.device
                ).cpu()
                # [num_samples, 3, H, W]->[num_samples, H, W, 3]
                #generated_imgs = [sample.numpy() for sample in samples.permute(0, 2, 3, 1)]
                generated_imgs = samples
                print("Finish generating")

            else: # test_loader is given
                generated_imgs = {'imgs': [], 'loss': [], 'file_name': [], 'label': []}
                for ii, batch in enumerate(test_loader):
                    img = batch['img'].to(self.device)
                    label = batch['label']
                    file_name = batch['file_name']

                    recon = self.model.generate(img)  
                    recons_loss = F.mse_loss(recon, img, reduction='none') 
                    generated_imgs['imgs'].append(recon.permute(0, 2, 3, 1).cpu()) # [B, C, H, W]->[B, H, W, C]
                    generated_imgs['loss'] += recons_loss.flatten().cpu().tolist() # [B, 1]
                    generated_imgs['label'] += label
                    generated_imgs['file_name'] += file_name             # list concatenation

                    print("Predicting {}/{} | mse_loss={}".format(
                        ii+1, len(test_loader), recons_loss.mean()), end='\r')

                generated_imgs['imgs'] = torch.cat(generated_imgs['imgs'], dim=0).numpy()
                print("Finish prediction")
        
        return generated_imgs

    def on_epoch_end(self):
        working_dir = os.path.join(
            "results", 
            f"VAE{self.model.hidden_size}_lr{self.learning_rate}_kldw_{self.kld_weight}",
            "VAE_exp"
        )
        
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        generated_imgs = self.predict(num_samples=32)  
        vutils.save_image(
            generated_imgs, 
            fp=os.path.join(working_dir, f"randomly_generated_epoch{self.epoch}.png"),
            normalize=True
        )