import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np

from .metric import MetricMeter
from .utils import calc_acc, wandb, Timer, TrainingLogger

mean = lambda x: sum(x)/len(x)

class DTNTrainer:

    def __init__(self, model, cfg=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        #self.device = 'cpu'
        #print("Torch seed: {}".format(torch.initial_seed()))


        # Training settings
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            [{'params': self.model.encoder.parameters(), 'lr':1e-6},
             {'params': self.model.hallucinator.parameters(), 'lr':1e-4}]
            #self.model.parameters()
            , lr=cfg['learning_rate'])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.num_epoch = cfg['epochs']
        self.train_way, self.val_way = cfg['train_way'], cfg['val_way']
        self.shot, self.query = cfg['shot'], cfg['query']
        self.cfg = cfg


        # Set the logger
        self.epoch = 1
        self.min_loss, self.best_acc = 10000.0, 0.0
        self.cfg_str = f"{self.val_way}way{self.shot}"
        
    
    def train(self, data_loaders, dest_dir):
        '''
        Train the model with `data_loaders`
        Args: 
            data_loaders (list of length 2): list containing train/val loader
            dest_dir: path to which metadata and log are dumped
        '''
        train_loader, val_loader, ref_loader = data_loaders
        timer = Timer()
        logger = TrainingLogger(dest_path=os.path.join(dest_dir, f"epoch_log_{self.cfg_str}.txt"))
        logger.print("Configs:\n"+"\n".join([f"{k}: {v}" for k, v in self.cfg.items()])+"\n", show=False)

        while self.epoch < self.num_epoch + 1: 

            self.model.train(True)
            loss_meter, acc_meter = MetricMeter(), MetricMeter()
            
            for i, (batch, batch_ref) in enumerate(zip(train_loader, ref_loader), start=1):
                
                # bs = way * shot + query
                img = batch['img'].to(self.device) # tensor: (bs, 3, 84, 84)
                label = batch['label'].to(self.device) # list: (bs)
                ref = batch_ref['img'].to(self.device) # tensor: (2M, 3, 84, 84)
                
                # shot = (shot*way, out_dim)
                # query = (query, out_dim)
                len_support = self.train_way * self.shot
                M = len(ref) // 2
                logits = self.model(
                    shot=img[:len_support], 
                    query=img[len_support:], 
                    ref1=ref[:M],
                    ref2=ref[M:],
                    num_way=self.train_way) # (query, way)                 
                label = make_label(label[:len_support], label[len_support:], self.train_way)
                
                loss = self.criterion(logits, label)
                acc, correct, total = calc_acc(logits, label, return_raw=True)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_meter.update(loss.item(), total=1)
                acc_meter.update(correct, total)
                              
                # if i % 10 == 0:
                #     print('Epoch {:3d} | ({:3d}/{:3d}) | Loss: {:5f} | Acc: {:5f}%' \
                #         .format(self.epoch, i, len(train_loader), 
                #         loss.item(), acc*100) + ' '*10, end='\r')
                    
                wandb.log({
                    'train_loss': loss.item(),
                    'train_acc': acc
                })
            
            logger.print('Epoch {:3d} | Train | Loss: {:5f} | Acc: {:5f}%'.format(self.epoch, 
                loss_meter.get_score(), acc_meter.get_score()*100) + ' '*10)


            ### Validation:
            val_loss, val_acc = self.evaluate(val_loader, ref_loader)    
            logger.print('Epoch {:3d} |  Val  | Loss: {:5f} | Acc: {:5f}%' \
                .format(self.epoch, val_loss, val_acc*100) + ' '*10)
            
            logger.print("Epoch time: {}/{}".format(timer.epoch_time(), 
                timer.measure(p=self.epoch/self.num_epoch)))


            ### Checkpointing:
            if val_loss < self.min_loss:
                self.min_loss = val_loss
                #torch.save(self.model.state_dict(), os.path.join(dest_dir, f"best_loss_{trg_loss:.4f}.pth"))
                self.save_ckpt(os.path.join(dest_dir, f"best_loss.pth"))
                logger.print("***Model saved.***")

            if val_acc > self.best_acc and abs(val_loss - self.min_loss) < 0.2:
                self.best_acc = val_acc
                #torch.save(self.model.state_dict(), os.path.join(dest_dir, f"best_loss_{trg_loss:.4f}.pth"))
                self.save_ckpt(os.path.join(dest_dir, f"best_acc.pth"))
                logger.print("***Model (acc) saved.***")

            self.epoch += 1 
            self.lr_scheduler.step()
            logger.print("\n")            
            
        os.rename(
            os.path.join(dest_dir, f"best_loss.pth"),
            os.path.join(dest_dir, f"best_loss_{self.min_loss:.5f}_{self.cfg_str}.pth")
        )
        os.rename(
            os.path.join(dest_dir, f"best_acc.pth"),
            os.path.join(dest_dir, f"best_acc_{self.best_acc*100:.3f}_{self.cfg_str}.pth")
        )

    def evaluate(self, test_loader, ref_loader, return_stats=True):
        loss_meter, acc_meter = MetricMeter(), MetricMeter()
        file_names = []
        predictions = []
        self.model.train(False)
        with torch.no_grad():
            
            for i, (batch, batch_ref) in enumerate(zip(test_loader, ref_loader), start=1):
                # bs = way * shot + query
                img = batch['img'].to(self.device) # tensor: (bs, 3, 84, 84)
                file_name = batch['file_name']
                ref = batch_ref['img'].to(self.device) # tensor: (2M, 3, 84, 84)

                # shot = (way*shot, out_dim)
                # query = (query, out_dim)
                len_support = self.val_way * self.shot
                M = len(ref) // 2
                logits = self.model(
                    shot=img[:len_support], 
                    query=img[len_support:], 
                    ref1=ref[:M],
                    ref2=ref[M:],
                    num_way=self.val_way) # (query, way) 

                predictions += logits.argmax(1).tolist()
                file_names += file_name      

                if any(batch['label']):  # any None in batch['label']
                    label = batch['label'].to(self.device) # [batch_size]                    
                    label = make_label(label[:len_support], label[len_support:], self.val_way)
                    
                    loss = self.criterion(logits, label)
                    acc, correct, total = calc_acc(logits, label, return_raw=True)
                    
                    loss_meter.update(loss.item(), total=1)
                    acc_meter.update(correct, total)
                    
                    # if i % 10 == 0: 
                    #     print('Epoch {:3d} ({:3d}/{:3d}) | Loss: {:5f} | Acc: {:5f}%' \
                    #     .format(self.epoch, i, len(test_loader), 
                    #     loss.item(), acc*100) + ' '*10, end='\r')

        if return_stats:
            return loss_meter.get_score(), acc_meter.get_score()
        else:
            return predictions, file_names
            
    def save_ckpt(self, path):
        # ckpt = {
        #     'model': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'min_loss': self.min_loss
        # }
        torch.save(self.model.state_dict(), path)

    def load_ckpt(self, path):
        self.model.load_state_dict(torch.load(path))



def make_label(label_s, label_q, num_way):
    '''
    Make one-hot labels.

    Args:
        label_s (tensor): shape=(num_way*shot, )
        label_q (tensor): shape=(query, )
        num_way (int): the number of classes
    Return:
        label: shape=(query, num_way)
            This can be thought of (batch_size, num_class)
    '''
    label_s = label_s[: num_way] # (num_way)
    label_q = label_q.view(-1, 1) # (query, 1)
    
    query = label_q.size(0)
    label = label_s.reshape(1, num_way).repeat(query, 1)  # (query, num_way)
    label = (label == label_q).long().argmax(1).reshape(-1) # (query, ), broadcasting label_q

    return label.long()