import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
#import wandb
from .logger import wandb

from .DANN import DANN, DANN_ResNet

torch.manual_seed(422)
def mean(x): return sum(x)/len(x)

wandb.init(project="dlcv_3-3")

class DANN_Config:
    def __init__(
        self,
        batch_size=32,
        epochs=1000,
        learning_rate=0.0001
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def get_cfg_dict(self):
        cfg = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }

        return cfg

class DANN_MODEL:
    def __init__(self, cfg=DANN_Config()):
        print("DANN model initalization.")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("***Torch seed:", torch.initial_seed())

        # Model init
        self.model = DANN()
        #self.model = DANN_ResNet()
        self.model.to(self.device)
        # Training settings
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size

        # Set the logger
        self.min_loss = 10000.0
    
    def train(self, train_loaders, val_loaders, dest_dir, domain_adaption=True):
        '''
        Train DANN with `train_loaders`

        Args: 
            train_loaders (dict): DataLoaders for training data in source/target domain
            val_loaders (dict): DataLoaders for val data in source/target domain
            dest_dir: path to which metadata and log are dumped
        '''
        len_dataloader = min(len(train_loaders['source']), len(train_loaders['target']))
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for ii, (src_batch, trg_batch) in enumerate(zip(train_loaders['source'], train_loaders['target'])):
                loss = 0.0
                p = float(ii + epoch * len_dataloader) / self.epochs / len_dataloader
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

                self.model.train()
                src_img = src_batch['image'].to(self.device)
                trg_img = trg_batch['image'].to(self.device)
                src_label = src_batch['label'].to(self.device)
                trg_label = trg_batch['label'].to(self.device)
                domain_labels = torch.zeros((src_img.size(0), 1), dtype=torch.float, device=self.device)
                # batch_size = src_img.size(0)

                pred_class, pred_domain = self.model(src_img, alpha)
                
                # Source part
                src_class_loss = self.class_criterion(pred_class, src_label)
                src_domain_loss = self.domain_criterion(pred_domain, domain_labels)

                trg_domain_loss = 0.0
                if domain_adaption:
                    # Target part
                    domain_labels = torch.ones((trg_img.size(0), 1), dtype=torch.float, device=self.device)
                    _, pred_domain = self.model(trg_img, alpha)
                    trg_domain_loss = self.domain_criterion(pred_domain, domain_labels)
                
                # Back Prop
                self.optimizer.zero_grad()
                batch_loss = src_class_loss + 1.5*src_domain_loss + 1.5*trg_domain_loss
                batch_loss.backward()
                self.optimizer.step()            

                print('Epoch {:3d} ({:3d}/{:3d}) | Total Loss: {:5f} | Loss(class/domain_src/domain_trg): {:.5f}/{:.5f}/{:.5f}' \
                      .format(epoch, ii+1, len_dataloader, batch_loss,
                              src_class_loss, src_class_loss, trg_domain_loss), end='\r')



                wandb.log({
                    'loss': batch_loss,
                    'src_class_loss': src_class_loss,
                    'src_class_loss': src_class_loss, 
                    'trg_domain_loss': trg_domain_loss
                })


            print('Train | Epoch {:3d} | Total Loss: {:5f} | Loss(class/domain_src/domain_trg): {:.5f}/{:.5f}/{:.5f}' \
                      .format(epoch, batch_loss, src_class_loss, src_class_loss, trg_domain_loss) + " "*10)

            src_acc, src_loss = self.evaluate(val_loaders['source'])
            trg_acc, trg_loss = self.evaluate(val_loaders['target'])
            print('Val   | Epoch {:3d} | Loss(src/trg): {:.4f}/{:.4f} | Acc(src/trg): {:.4f}/{:.4f} \n' \
                      .format(epoch, src_loss, trg_loss, src_acc*100, trg_acc*100) + " "*10)
            
            wandb.log({
                    'trg_acc': trg_acc,
                    'src_acc': src_acc,
                    'src_loss': src_loss, 
                    'trg_loss': trg_loss,
            })

            if trg_loss < self.min_loss:
                self.min_loss = trg_loss
                #torch.save(self.model.state_dict(), os.path.join(dest_dir, f"best_loss_{trg_loss:.4f}.pth"))
                torch.save(self.model.state_dict(), os.path.join(dest_dir, f"best_loss.pth"))
                print("***Model saved.***")

    def evaluate(self, test_loader, dump=False, print_std=False):
        
        total_acc = MetricMeter()
        total_loss = MetricMeter()
        self.model.eval()
        with torch.no_grad():
            for ii, batch in enumerate(test_loader):
                images = batch['image'].to(self.device)
                file_id = batch['file_id']
                preds, _ = self.model(images, alpha=0) 

                if batch['label'] is not None: 
                    labels = batch['label'].to(self.device) # [batch_size]

                    # calculate loss
                    batch_loss = self.class_criterion(preds, labels).item()
                    total_loss.update(batch_loss, labels.size(0))


                    preds = preds.argmax(1) # [batch_size]
                    correct = torch.sum(preds == labels).item()
                    total_acc.update(correct, labels.size(0))

                    if print_std:
                        print("Evaluating {:3d}/{:3d}: acc={:.4f} | loss={:.4f}" \
                            .format(ii, len(test_loader), total_acc.get_score()*100, total_loss.get_score()), end='\r')

        if print_std:
            print("Acc={:.4f} | loss={:.4f}" \
                    .format(total_acc.get_score()*100, total_loss.get_score()), end='\n')

        return total_acc.get_score(), total_loss.get_score()

    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def load_ckpt(self, path):
        self.model.load_state_dict(torch.load(path))

class MetricMeter():
    def __init__(self, at=1):
        self.at = at
        self.n = 0
        self.n_corrects = 0
        self.name = 'Accuracy@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, correct, total):        
        self.n += total
        self.n_corrects += correct

    def get_score(self):
        return self.n_corrects / self.n if self.n != 0 else 0.0
