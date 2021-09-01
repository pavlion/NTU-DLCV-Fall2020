import os
import time
import torch
import torch.utils.data.dataloader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
import numpy as np

from utils import TrainingLogger, Accuracy
from loss import lovasz_softmax

class Trainer():

    def __init__(self,
                 max_epochs=10,
                 device=None,
                 learning_rate=1e-3,
                 optimizer_opt='Adam',
                 loss_opt='CELoss',
                 scheduler_opt=None, 
                 max_iters_in_epoch=1e20,
                 eval_per_epoch=False,
                 model=None,
                 metrics=[],
                 train_loader=None,
                 val_loader=None):

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch
        self.eval_per_epoch = eval_per_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model
        self.optimizer = {
            'AdamW': torch.optim.AdamW(self.model.parameters(),
                                       lr=self.learning_rate),
            'Adam':  torch.optim.Adam(self.model.parameters(),
                                      lr=self.learning_rate),
            'SGD':   torch.optim.SGD(self.model.parameters(),
                                     lr=self.learning_rate)
        }.get(optimizer_opt, torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate))

        self.criterion = {
            'CELoss':               torch.nn.CrossEntropyLoss(),
            'Lovasz':               lovasz_softmax
        }.get(loss_opt, torch.nn.CrossEntropyLoss())
        
        if scheduler_opt == 'StepLR':
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.97)
        else:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=1)
        
        # Logger
        self.train_logger = TrainingLogger(max_epochs)
        self.AccuracyMeter = Accuracy()

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.model.to(self.device)
        print(f"***Device: {self.device}")  
        #self.optimizer = self.optimizer.to(device)  
        self.epoch = 0
        self.min_loss = 100000.0
        self.best_metrics = None
        
    def train(self, ckpt_path, callbacks=[]):
        
        self.train_logger.reset(dest_dir=ckpt_path)
        min_loss = 100000.0
        miou = 0.0
        while self.epoch < self.max_epochs:
            
            self.train_logger.print_and_update(f"Epoch {self.epoch}")
            log_train = self._run_epoch(self.train_loader, is_training=True)
            log_valid = self._run_epoch(self.val_loader, is_training=False)
            self.lr_scheduler.step()

            epoch_msg = self.epoch_logger(log_train, log_valid)
            self.train_logger.print_and_update(epoch_msg, epoch_end=True)

            if log_valid['loss'] < 0.35 and log_valid['miou'] > 0.70:
                loss = log_valid['loss']
                miou = log_valid['miou']
                model_path = os.path.join(ckpt_path, f"best_loss{loss:.6f}_miou{miou:.6f}_{self.model.__name__}.ckpt")
                self.save(model_path)
                self.train_logger.print_and_update("***Model saved (special)***")

            if self.epoch == 0 or self.epoch == 3 or self.epoch == 5:
                loss = log_valid['loss']
                miou = log_valid['miou']
                model_path = os.path.join(
                    ckpt_path, 
                    f"{self.model.__name__}_epoch{self.epoch}_loss{loss:.6f}_miou{miou:.6f}.ckpt"
                )
                self.save(model_path)


            if log_valid['loss'] < self.min_loss:
                self.min_loss = log_valid['loss']
                self.best_metrics = log_valid
                miou = log_valid['miou']
                model_path = os.path.join(ckpt_path, f"best_loss_{self.model.__name__}.ckpt")
                self.save(model_path)
                self.train_logger.print_and_update("***Model saved***")

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)
            
            self.epoch += 1
        
        # change the name of the best model 
        os.rename(
            os.path.join(ckpt_path, f"best_loss_{self.model.__name__}.ckpt"),
            os.path.join(ckpt_path, f"{self.model.__name__}_loss{self.min_loss:.6f}_miou{miou:.6f}.ckpt")
        )
        self.train_logger.dump_logs()

    def _run_epoch(self, dataloader, is_training):

        description = 'Train' if is_training else 'Val'

        self.AccuracyMeter.reset()
        loss = 0
        for i, batch in enumerate(dataloader):

            if is_training:
                self.model.train(True)

                # Forward propagation
                seq = batch['img'].to(self.device)
                label = batch['label'].long().to(self.device)
                logits = self.model(seq) # [bs, 7, W, H]

                # Backward propagation
                self.optimizer.zero_grad()
                batch_loss = self.criterion(logits, label)
                batch_loss.backward()
                self.optimizer.step(batch_loss)

            else:
                self.model.train(False)
                with torch.no_grad():
                    seq = batch['img'].to(self.device)
                    label = batch['label'].long().to(self.device)
                    logits = self.model(seq)  # [batch, out_dim]
                    batch_loss = self.criterion(logits, label)

            loss += batch_loss.item()
            acc = self.update_acc(logits, label)
            print("{:5s} | Steps:{:4d}/{:-4d}  |  loss={:.6f}  |  acc={:.2f}%"
                  .format(description, i, len(dataloader), loss/(i+1), acc*100.0), end='\r')

        loss /= len(dataloader)
        acc = self.AccuracyMeter.get_score()

        epoch_log = {
            'loss': float(loss),
            'acc': float(acc)
        }

        return epoch_log

    def update_acc(self, pred, label):
        total_num = pred.size(0)
        correct = (pred.argmax(1) == label).int().sum().cpu().item()
        self.AccuracyMeter.update(correct, total_num)

        return self.AccuracyMeter.get_score()

    def predict(self, test_loader):
        self.model.eval()
        ys_ = []
        loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                with torch.no_grad():
                    seq = batch['img'].to(self.device)
                    label = batch['label'].long().to(self.device)
                    logits = self.model(seq)  # [batch, out_dim]
                    batch_loss = self.criterion(logits, label)
                
                loss += float(batch_loss.item())
                ys_.append(logits)
                print("Predicting {}/{} | loss={:.4f}".format(i+1, len(test_loader), loss/(i+1)), end='\r')
        print("Finish predicting loss={:.4f}".format(loss/len(test_loader))+' '*50, end='\n')
        
        ys_ = torch.cat(ys_, 0)
        return ys_

    def epoch_logger(self, log_train, log_valid):
        metrics_log = ' | '.join([f'val {name}={value}' for name, value in log_valid.items()])
        msg = "train loss={:.6f} | val loss={:.6f} |  {}"\
            .format(log_train['loss'], log_valid['loss'], metrics_log)+" "*50
        
        return msg

    def save(self, path):
        torch.save({
            #'epoch': self.epoch + 1,
            'min_loss': self.min_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        #self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        #self.min_loss = checkpoint['min_loss']
        print("Checkpoint loaded.")

class P2Trainer(Trainer):    

    def __init__(self,
                 max_epochs=10,
                 device=None,
                 learning_rate=1e-3,
                 optimizer_opt='AdamW',
                 loss_opt='CELoss',
                 max_iters_in_epoch=1e20,
                 eval_per_epoch=False,
                 scheduler_opt=None,
                 model=None,
                 metrics=[],
                 train_loader=None,
                 val_loader=None):
                 
        super().__init__(max_epochs=max_epochs,
                 device=device,
                 learning_rate=learning_rate,
                 optimizer_opt=optimizer_opt,
                 loss_opt=optimizer_opt,
                 scheduler_opt=scheduler_opt,
                 model=model,
                 train_loader=train_loader,
                 val_loader=val_loader)
                 
        print("Torch seed:", torch.initial_seed())   
        #self.optimizer = torch.optim.Adam(
        #    [{'params': model.backbone.parameters(), 'lr': 3e-5},
        #     {'params': model.conv_block.parameters(), 'lr': 3e-4}
        #])

    def _run_epoch(self, dataloader, is_training):

        description = 'Train' if is_training else 'Val'

        loss = 0
        preds, labels = [], []
        for i, batch in enumerate(dataloader):

            if is_training:
                self.model.train(True)

                # Forward propagation
                seq = batch['img'].to(self.device)
                label = batch['label'].long().to(self.device) # [bs,W, H]
                logit = self.model(seq) # [bs, 7, W, H]

                # Backward propagation
                self.optimizer.zero_grad()
                batch_loss = self.criterion(logit, label)
                batch_loss.backward()
                self.optimizer.step()

            else:
                self.model.train(False)
                with torch.no_grad():
                    seq = batch['img'].to(self.device)
                    label = batch['label'].long().to(self.device)
                    logit = self.model(seq)  
                    batch_loss = self.criterion(logit, label)
                    
            loss += batch_loss.item()
            #miou = 0.0
            miou = self.calc_miou(logit.argmax(1).cpu(), label)
            print("{:5s} | Steps:{:4d}/{:-4d}  |  loss={:.6f}  |  miou={:.2f}%"
                  .format(description, i, len(dataloader), loss/(i+1), miou*100.0), end='\r')
            
            preds.append(logit.argmax(1).cpu())
            labels.append(label.cpu())   

        loss /= len(dataloader)
        #miou = 0.0
        to_tensor = lambda x: torch.cat(x, dim=0)
        miou = self.calc_miou(to_tensor(preds), to_tensor(labels))

        epoch_log = {
            'loss': float(loss),
            'miou': float(miou)
        }

        return epoch_log

    def calc_miou(self, preds, labels):
        preds = preds.cpu().numpy()   # [bs*n, W, H] (n batches)
        labels = labels.cpu().numpy() # [bs*n, W, H] (n batches)

        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(preds == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((preds == i) * (labels == i))
            iou = tp / (tp_fp + tp_fn - tp) if (tp_fp + tp_fn - tp) != 0 else 0.0
            mean_iou += iou / 6

        return mean_iou

    def epoch_logger(self, log_train, log_valid):
        '''
        Message to be printed after each epoch. (Override parent method.)
        '''
        msg = "train loss={:.6f} | val loss={:.6f} | val miou={:.6f}%".format(log_train['loss'], \
            log_valid['loss'], log_valid['miou']*100.0) + " "*50
        
        return msg
        
    def predict(self, test_loader, dest_dir, no_label=True):
        loss = 0
        self.model.eval()
        predictions = {'id': [], 'pred': []}
        preds_epoch = []
        labels_epoch = []
        with torch.no_grad():
            for ii, batch in enumerate(test_loader):
                seq = batch['img'].to(self.device)
                ids = batch['id']
                logits = self.model(seq)  # [batch, out_dim]
                pred = logits.argmax(1).cpu() # [batch, 7, H, W] -> [batch, H, W]

                if not no_label:
                    label = batch['label']
                    
                    # calculate loss
                    batch_loss = self.criterion(logits, label.to(self.device))
                    loss += batch_loss.item()

                    # calculate miou
                    miou = self.calc_miou(pred, label)
                    
                    # collect all the predictions and labels
                    preds_epoch.append(pred)
                    labels_epoch.append(label)
                    print("Predicting {}/{}, loss={:.6f}, miou={:.6f}".format( \
                        ii+1, len(test_loader), loss/(ii+1), miou), end='\r')
                else:
                    print("Predicting {}/{}".format(ii+1, len(test_loader)), end='\r')

                # list concatenation
                predictions['id'] += ids
                predictions['pred'] += pred.tolist()

            if no_label:
                print("Finish prediction.")
            else:
                preds = torch.cat(preds_epoch, dim=0)
                labels = torch.cat(labels_epoch, dim=0)
                miou = self.calc_miou(preds, labels)
                print("Finish prediction: loss={:.6f}, miou={:.6f}".format(loss/len(test_loader), miou))

        return predictions
        
        
class P1Trainer(Trainer):

    def __init__(self,
                 max_epochs=10,
                 device=None,
                 learning_rate=1e-3,
                 optimizer_opt='AdamW',
                 loss_opt='CELoss',
                 max_iters_in_epoch=1e20,
                 eval_per_epoch=False,
                 scheduler_opt=None,
                 model=None,
                 metrics=[],
                 train_loader=None,
                 val_loader=None):
                 
        super().__init__(max_epochs=max_epochs,
                 device=device,
                 learning_rate=learning_rate,
                 optimizer_opt=optimizer_opt,
                 loss_opt=optimizer_opt,
                 scheduler_opt=scheduler_opt,
                 model=model,
                 train_loader=train_loader,
                 val_loader=val_loader)
                 
        print("Torch seed:", torch.initial_seed())   
        print("Model name:", self.model.__name__)   
        #self.optimizer = torch.optim.Adam(
        #    [{'params': model.backbone.parameters(), 'lr': 3e-5},
        #     {'params': model.conv_block.parameters(), 'lr': 3e-4}
        #])

        
    def train(self, ckpt_path, callbacks=[]):
        
        self.train_logger.reset()
        min_loss = 100000.0
        acc = 0.0
        while self.epoch < self.max_epochs:
            self.train_logger.print_and_update(f"Epoch {self.epoch}")
            log_train = self._run_epoch(self.train_loader, is_training=True)
            log_valid = self._run_epoch(self.val_loader, is_training=False)
            self.lr_scheduler.step()

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)

            epoch_msg = self.epoch_logger(log_train, log_valid)
            self.train_logger.print_and_update(epoch_msg, epoch_end=True)

            if log_valid['loss'] < self.min_loss:
                self.min_loss = log_valid['loss']
                self.best_metrics = log_valid
                acc = log_valid['acc'] * 100.0 
                model_path = os.path.join(
                    ckpt_path, 
                    f"best_loss_{self.model.__name__}.ckpt"
                )
                self.save(model_path)
                self.train_logger.print_and_update("***Model saved***")

            self.epoch += 1
        
        # change the name of the best model 
        os.rename(
            os.path.join(ckpt_path, f"best_loss_{self.model.__name__}.ckpt"),
            os.path.join(ckpt_path, f"{self.model.__name__}_loss{self.min_loss:.6f}_acc{acc:.6f}.ckpt")
        )
        self.train_logger.dump_logs()

    def _run_epoch(self, dataloader, is_training):

        description = 'Train' if is_training else 'Val'

        self.AccuracyMeter.reset()
        loss = 0
        for i, batch in enumerate(dataloader):

            if is_training:
                self.model.train(True)

                # Forward propagation
                seq = batch['img'].to(self.device)
                label = batch['label'].long().to(self.device)
                logits = self.model(seq) # [bs, 7, W, H]

                # Backward propagation
                self.optimizer.zero_grad()
                batch_loss = self.criterion(logits, label)
                batch_loss.backward()
                self.optimizer.step()

            else:
                self.model.train(False)
                with torch.no_grad():
                    seq = batch['img'].to(self.device)
                    label = batch['label'].long().to(self.device)
                    logits = self.model(seq)  # [batch, out_dim]
                    batch_loss = self.criterion(logits, label)

            loss += batch_loss.item()
            acc = self.update_acc(logits, label)
            print("{:5s} | Steps:{:4d}/{:-4d}  |  loss={:.6f}  |  acc={:.2f}%"
                  .format(description, i, len(dataloader), loss/(i+1), acc*100.0)+" "*50, end='\r')

        loss /= len(dataloader)
        acc = self.AccuracyMeter.get_score()

        epoch_log = {
            'loss': float(loss),
            'acc': float(acc)
        }

        return epoch_log

    def update_acc(self, pred, label):
        total_num = pred.size(0)
        correct = (pred == label).int().sum().cpu().item()
        self.AccuracyMeter.update(correct, total_num)

        return self.AccuracyMeter.get_score()

    def predict(self, test_loader, no_label=True):
        acc_meter = Accuracy()
        self.model.eval()
        predictions = {'id': [], 'pred': []}
        loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                seq = batch['img'].to(self.device)
                label = batch['label'].long().to(self.device)
                id = batch['id']
                logits = self.model(seq)  # [batch, out_dim]
                preds = logits.argmax(1)
                
                predictions['id'] += id # id is a list
                predictions['pred'] += preds.tolist() # preds: [batch]
                
                # Compute metric if label is provided
                if not no_label:
                    batch_loss = self.criterion(logits, label)
                    loss += float(batch_loss.item())
                
                    acc_meter.update(
                        (preds == label).int().sum().cpu().item(), # correct
                        logits.size(0) # total
                    )
                    acc = acc_meter.get_score() * 100.0

                    print("Predicting {}/{} | loss={:.6f} | acc={:.6f}%".format(
                        i+1, len(test_loader), loss/(i+1), acc), end='\r')
                                    
                else:
                    print("Predicting {}/{}".format(i+1, len(test_loader)), end='\r')

        if not no_label:
            print("Finish predicting loss={:.6f}, acc={:.6f}%".format(
                loss/len(test_loader), acc_meter.get_score()*100.0)+' '*50, end='\n')
        else: 
            print("Finish predicting"+' '*50, end='\n')
                
        
        return predictions

    def epoch_logger(self, log_train, log_valid):
        msg = "train loss={:.6f} | val loss={:.6f} |  val acc={:.4f}%"\
            .format(log_train['loss'], log_valid['loss'], log_valid['acc']*100.0)+" "*50
        
        return msg


if __name__ == "__main__":
    pass