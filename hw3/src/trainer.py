import os
import time
import torch
from torch.optim import lr_scheduler
import numpy as np

from utils import TrainingLogger, Accuracy


class Trainer():

    def __init__(self,
                 max_epochs=10,
                 device=None,
                 learning_rate=1e-3,
                 optimizer_opt='Adam',
                 loss_opt='CELoss',
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
        }.get(loss_opt, torch.nn.CrossEntropyLoss())
        
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
        
        # Logging-related info
        self.epoch = 0
        self.saved_model_name = f"best_loss_{type(self.model).__name__}"
        self.min_loss = 100000.0
        self.best_metrics = None
        
    def train(self, ckpt_path, callbacks=[]):
        
        self.train_logger.reset(dest_dir=ckpt_path)
        while self.epoch < self.max_epochs:
            
            self.train_logger.print_and_update(f"Epoch {self.epoch}")
            log_train = self._run_epoch(self.train_loader, is_training=True)
            log_valid = self._run_epoch(self.val_loader, is_training=False)

            epoch_msg = self.epoch_logger(log_train, log_valid)
            self.train_logger.print_and_update(epoch_msg, epoch_end=True)

            if log_valid['loss'] < self.min_loss:
                self.min_loss = log_valid['loss']
                self.best_metrics = log_valid
                model_path = os.path.join(ckpt_path, f"best_loss.pth")
                self.save(model_path)
                self.train_logger.print_and_update("***Model saved***")
            
            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)
            
            self.on_epoch_end()
            self.epoch += 1
        os.rename(
            os.path.join(ckpt_path, f"best_loss.pth"), 
            os.path.join(ckpt_path, self.saved_model_name)
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

    def on_epoch_end(self):
        pass

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

if __name__ == "__main__":
    pass