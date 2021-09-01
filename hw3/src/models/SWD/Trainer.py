import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
#import wandb
from ..logger import wandb

from .SWD import SWD, discrepancy_mcd, discrepancy_slice_wasserstein

def mean(x): return sum(x)/len(x)

wandb.init(project="dlcv_3-4_SWD")

class SWD_Config:
    def __init__(
        self,
        batch_size=64,
        epochs=50,
        num_filters=64,
        learning_rate=0.005,
        M=128, 
        adapt_loss_opt='SWD',
        src_domain='usps',
        trg_domain='mnistm'
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.adapt_loss_opt = adapt_loss_opt if adapt_loss_opt in ('SWD', 'MCD') else 'SWD'
        self.M = M
        self.src_domain = src_domain
        self.trg_domain = trg_domain

    def get_cfg_dict(self):        
        cfg = {
            'batch_size': self.batch_size,
            'num_filters': self.num_filters,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'adapt_loss_opt': self.adapt_loss_opt,
            'M': self.M,
            'src_domain': self.src_domain,
            'trg_domain': self.trg_domain
        }

        return cfg

class SWD_Trainer:

    def __init__(self, cfg=SWD_Config()):
        print("SWD model initalization.")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("***Torch seed: {}***".format(torch.initial_seed()))

        # Model init
        if cfg.src_domain != 'usps':
            self.G = SWD.Generator(cfg.num_filters).to(self.device)
            self.C1 = SWD.Classifier().to(self.device)
            self.C2 = SWD.Classifier().to(self.device)
        else: 
            print("USPS model is used.")
            self.G = SWD.USPS.Generator().to(self.device)
            self.C1 = SWD.USPS.Classifier().to(self.device)
            self.C2 = SWD.USPS.Classifier().to(self.device)

        # Training settings
        self.criterion = nn.CrossEntropyLoss()
        self.discrepancy_loss = {
            'SWD': discrepancy_slice_wasserstein(M=128),
            'MCD': discrepancy_mcd
        }[cfg.adapt_loss_opt]

        self.optimizer_G = optim.Adam(self.G.parameters(), lr=cfg.learning_rate)
        self.optimizer_F = optim.Adam(
            list(self.C1.parameters())+list(self.C2.parameters()), 
            lr=cfg.learning_rate
        )
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size

        # Set the logger
        self.min_loss = 10000.0
        self.max_acc = 0.0
    
    def train(self, train_loaders, val_loaders, dest_dir, domain_adaption=True):
        '''
        Train SWD with `train_loaders`

        Args: 
            train_loaders (dict): DataLoaders for training data in source/target domain
            val_loaders (dict): DataLoaders for val data in source/target domain
            dest_dir: path to which metadata and log are dumped
        '''
        len_dataloader = min(len(train_loaders['source']), len(train_loaders['target']))
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for ii, (src_batch, trg_batch) in enumerate(zip(train_loaders['source'], train_loaders['target'])):
                
                self.set_model_mode(is_train=True)

                src_img = src_batch['image'].to(self.device)
                trg_img = trg_batch['image'].to(self.device)
                src_label = src_batch['label'].to(self.device)
                trg_label = trg_batch['label'].to(self.device)
                
                # Algorithm1, Step1: Train G, C1, C2 on labeled src data
                feature = self.G(src_img)
                pred1 = self.C1(feature)
                pred2 = self.C2(feature)
                loss_source = self.criterion(pred1, src_label) + self.criterion(pred2, src_label)

                self.optimizer_G.zero_grad(); self.optimizer_F.zero_grad()
                loss_source.backward()
                self.optimizer_F.step(); self.optimizer_G.step()
                

                if not domain_adaption:
                    continue
                    
                # Algorithm1, Step2: Train C1, C2 to maximize SWD on unlabled trg data
                loss_swd = 0.0
                src_feature = self.G(src_img).detach()
                src_pred1 = self.C1(src_feature)
                src_pred2 = self.C2(src_feature)
                loss_swd += self.criterion(src_pred1, src_label) + self.criterion(src_pred2, src_label)
                
                trg_feature = self.G(trg_img).detach()
                trg_pred1 = self.C1(trg_feature)
                trg_pred2 = self.C2(trg_feature)

                loss_swd -= 1*self.discrepancy_loss(trg_pred1, trg_pred2)

                self.optimizer_G.zero_grad(); self.optimizer_F.zero_grad()
                loss_swd.backward()
                self.optimizer_F.step()

                
                # Algorithm1, Step3: Train G to minimized SWD on unlabled trg data                
                for i in range(10):
                    trg_feature = self.G(trg_img)
                    trg_pred1 = self.C1(trg_feature)
                    trg_pred2 = self.C2(trg_feature)

                    loss_dis = self.discrepancy_loss(trg_pred1, trg_pred2)

                    self.optimizer_G.zero_grad(); self.optimizer_F.zero_grad()
                    loss_dis.backward()
                    self.optimizer_G.step()
                
                if ii % 10 == 0:
                    print('Epoch {:3d} ({:3d}/{:3d}) | Loss(src/swd/dis): {:5f} / {:.5f} / {:.5f}' \
                        .format(epoch, ii+1, len_dataloader, loss_source, loss_swd, loss_dis), end='\r')
                    
                epoch_loss += loss_source

                wandb.log({
                    'class_loss': loss_source
                })


            print('Train | Epoch {:3d} | Loss(src/swd/dis): {:5f} / {:.5f} / {:.5f}' \
                    .format(epoch, loss_source, loss_swd, loss_dis))
                

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
                torch.save(self.G.state_dict(), os.path.join(dest_dir, f"best_loss_G.pth"))
                torch.save(self.C1.state_dict(), os.path.join(dest_dir, f"best_loss_C1.pth"))
                torch.save(self.C2.state_dict(), os.path.join(dest_dir, f"best_loss_C2.pth"))
                print("***Loss Model Saved.***")
            
            if trg_acc > self.max_acc:
                self.max_acc = trg_acc
                #torch.save(self.model.state_dict(), os.path.join(dest_dir, f"best_loss_{trg_loss:.4f}.pth"))
                torch.save(self.G.state_dict(), os.path.join(dest_dir, f"best_acc_G.pth"))
                torch.save(self.C1.state_dict(), os.path.join(dest_dir, f"best_acc_C1.pth"))
                torch.save(self.C2.state_dict(), os.path.join(dest_dir, f"best_acc_C2.pth"))
                print("***Acc Model Saved.***")

    def evaluate(self, test_loader, ensemble=True, print_std=False):
        total_acc = MetricMeter()
        total_loss = MetricMeter()

        has_label = False
        file_names = []
        predictions = []
        self.set_model_mode(is_train=False)
        with torch.no_grad():
            for ii, batch in enumerate(test_loader):
                images = batch['image'].to(self.device)
                file_id = batch['file_id']
                features = self.G(images) 
                preds1 = self.C1(features) # [bs, 10]
                preds2 = self.C2(features) # [bs, 10]

                pred_ensemble = preds1 + preds2 if ensemble else preds2
                

                file_names += file_id
                predictions += pred_ensemble.argmax(1) # [bs]               



                if batch['label'] is not None: 
                    labels = batch['label'].to(self.device) # [batch_size]
                    # calculate loss
                    batch_loss = self.criterion(pred_ensemble, labels).item()
                    correct = torch.sum(pred_ensemble.argmax(1) == labels).item()

                    total_acc.update(correct, labels.size(0))
                    total_loss.update(batch_loss, labels.size(0))

                    print("Evaluating {:3d}/{:3d}: acc={:.4f} | loss={:.4f}" \
                            .format(ii, len(test_loader), total_acc.get_score()*100, total_loss.get_score()), end='\r')

        if print_std:
            print("Acc={:.4f}% | loss={:.4f}".format(total_acc.get_score()*100, 
                total_loss.get_score()) + " "*10, end='\n')

        return total_acc.get_score(), total_loss.get_score()

    def set_model_mode(self, is_train=True):
        self.G.train(is_train)
        self.C1.train(is_train)
        self.C2.train(is_train)

    def save_ckpt(self, path_G, path_C1, path_C2):
        torch.save(self.G.state_dict(), path_G)
        torch.save(self.C1.state_dict(), path_C1)
        torch.save(self.C2.state_dict(), path_C2)


    def load_ckpt(self, path_G, path_C1, path_C2):
        self.G.load_state_dict(torch.load(path_G))
        self.C1.load_state_dict(torch.load(path_C1))
        self.C2.load_state_dict(torch.load(path_C2))


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

class Logger():
    def __init__(self, targets):
        log = {}
        for target in targets:
            log[target] = []
        
        self.log = log

    def update(self, logs):
        if type(logs) != 'dict':
            return
        
        for k, v in logs.items():
            self.log[k].append(v)
    
    def get_mean(self):

        mean = {}
        for k in self.log.keys():
            mean[k] = sum(self.log[k])/len(self.log[k])
        
        return mean
