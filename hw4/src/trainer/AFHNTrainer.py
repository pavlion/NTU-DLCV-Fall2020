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

wandb.init(project="dlcv_4-3_AFHN")

class AFHNTrainer:

    def __init__(self, models, cfg=None):
        #print("ProtoNet model initalization.")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        #print("Torch seed: {}".format(torch.initial_seed()))


        # Training settings
        classifier, G, D = models
        self.classifier, self.G, self.D = classifier.to(self.device), \
                                    G.to(self.device), D.to(self.device)
        F.cross_entropy()                            
        self.criterion = nn.CrossEntropyLoss()

        self.optim_G = optim.Adam(self.model.parameters(), lr=cfg['learning_rate'])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.num_epoch = cfg['epochs']
        self.train_way, self.val_way = cfg['train_way'], cfg['val_way']
        self.shot, self.query = cfg['shot'], cfg['query']


        # Set the logger
        self.epoch = 1
        self.min_loss, self.best_acc = 10000.0, 0.0
        self.cfg_str = f"{self.val_way}way{self.shot}shot_M{cfg['M']}_fc{cfg['fc_dim']}"
        self.logger = TrainingLogger(dest_path=os.path.join("ckpt", f"epoch_log_{self.cfg_str}.txt"))
        self.logger.print(
            "Configs:\n"+"\n".join([f"{k}: {v}" for k, v in cfg.items()])+"\n",
            show=False
        )
    
    def train(self, data_loaders, dest_dir):
        '''
        Train the model with `data_loaders`
        Args: 
            data_loaders (list of length 2): list containing train/val loader
            dest_dir: path to which metadata and log are dumped
        '''
        train_loader, val_loader = data_loaders
        timer = Timer()

        while self.epoch < self.num_epoch + 1: 

            self.model.train(True)
            loss_meter, acc_meter = MetricMeter(), MetricMeter()
            
            for i, batch in enumerate(train_loader, start=1):
                
                # bs = way * shot + query
                img = batch['img'].to(self.device) # tensor: (bs, 3, 84, 84)
                label = batch['label'].to(self.device) # list: (bs)
                label = make_label(label[:len_support], label[len_support:], self.train_way)
                
                # shot = (shot*way, out_dim)
                # query = (query, out_dim)
                len_support = self.train_way * self.shot
                shot, query = img[:len_support], img[len_support:]
                logits, s, s_tilda, noise = self.model(shot, query, num_way=self.train_way) # (query, way)                 
                loss_cls = F.cross_entropy(logits, label)
                
                # s, s_tilda, tilda = (m, way, hid), m=1
                s, s_tilda, noise = s.squeeze(0), s_tilda.squeeze(0), noise.squeeze(0)
                real, fake = self.D(s, noise), self.D(s_tilda, noise)
                loss_gan = torch.mean(fake.detach()) - torch.mean(real)

                # Fix G and update D
                self.optimizer_D.zero_grad()
                loss = loss_cls + loss_gan
                loss.backward()
                self.optimizer_D.step()

                
                # Fix D and update G
                logits, s, s_tilda, noise = self.model(shot, query, num_way=self.train_way) # (query, way)                 
                loss_cls = F.cross_entropy(logits, label)
                
                # s, s_tilda, tilda = (m, way, hid), m=1
                s, s_tilda, noise = s.squeeze(0), s_tilda.squeeze(0), noise.squeeze(0)
                real, fake = self.D(s, noise), self.D(s_tilda, noise)
                loss_gan = torch.mean(fake.detach()) - torch.mean(real)



                
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
            
            self.logger.print('Epoch {:3d} | Train | Loss: {:5f} | Acc: {:5f}%'.format(self.epoch, 
                loss_meter.get_score(), acc_meter.get_score()*100) + ' '*10)


            ### Validation:
            val_loss, val_acc = self.evaluate(val_loader)    
            self.logger.print('Epoch {:3d} |  Val  | Loss: {:5f} | Acc: {:5f}%' \
                .format(self.epoch, val_loss, val_acc*100) + ' '*10)
            
            self.logger.print("Epoch time: {}/{}".format(timer.epoch_time(), 
                timer.measure(p=self.epoch/self.num_epoch)))


            ### Checkpointing:
            if val_loss < self.min_loss:
                self.min_loss = val_loss
                #torch.save(self.model.state_dict(), os.path.join(dest_dir, f"best_loss_{trg_loss:.4f}.pth"))
                self.save_ckpt(os.path.join(dest_dir, f"best_loss.pth"))
                self.logger.print("***Model saved.***")

            if val_acc > self.best_acc and abs(val_loss - self.min_loss) < 0.2:
                self.best_acc = val_acc
                #torch.save(self.model.state_dict(), os.path.join(dest_dir, f"best_loss_{trg_loss:.4f}.pth"))
                self.save_ckpt(os.path.join(dest_dir, f"best_acc.pth"))
                self.logger.print("***Model (acc) saved.***")

            self.epoch += 1 
            self.lr_scheduler.step()
            self.logger.print("\n")            
            
        os.rename(
            os.path.join(dest_dir, f"best_loss.pth"),
            os.path.join(dest_dir, f"best_loss_{self.min_loss:.5f}_{self.cfg_str}.pth")
        )
        os.rename(
            os.path.join(dest_dir, f"best_acc.pth"),
            os.path.join(dest_dir, f"best_acc_{self.best_acc*100:.3f}_{self.cfg_str}.pth")
        )

    def evaluate(self, test_loader, return_stats=True):
        loss_meter, acc_meter = MetricMeter(), MetricMeter()
        file_names = []
        predictions = []
        self.model.train(False)
        with torch.no_grad():
            for i, batch in enumerate(test_loader, start=1):
                len_support = self.val_way * self.shot
                # bs = way * shot + query
                img = batch['img'].to(self.device) # tensor: (bs, 3, 84, 84)
                file_name = batch['file_name']

                # shot = (way*shot, out_dim)
                # query = (query, out_dim)
                logits = self.model(img[:len_support], img[len_support:], num_way=self.val_way) # (query, way)  

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