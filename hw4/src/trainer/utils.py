import os 
import time
import random
import torch.nn as nn

try:
    import wandb
except ModuleNotFoundError:
    class wandb_fake:
        def init(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass
        
        def Image(self, *args, **kwargs):
            pass    
    #print("Fake wandb is used.")
    wandb = wandb_fake()

def calc_acc(pred, label, return_raw=False):
    '''
    Calculate the accuracy
    Args:
        pred  (tensor): shape=(batch_size, num_class)
        label (tensor): shape=(batch_size)
    '''
    pred_idx = pred.argmax(dim=1) #
    correct = (pred_idx == label).int().sum().item()
    total = len(label)

    if return_raw: 
        return correct/total, correct, total
        
    return correct/total


class Timer():

    def __init__(self):
        self.o = time.time()
        self.last_time = self.o

    def epoch_time(self):
        curr_time = time.time()
        duration = curr_time - self.last_time
        self.last_time = time.time()

        return '{:.2f}s'.format(duration)

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))

        return '{:.2f}s'.format(x)

class TrainingLogger:

    def __init__(self, dest_path='.'):
        self.epoch = 0
        self.logs = []
        self.dest_path = dest_path 
    
    def reset(self, dest_path='.'):
        self.logs = [] 
        self.epoch = 0
        self.dest_path = dest_path 

    def print(self, epoch_msg, show=True, epoch_end=False):
        if show:
            print(epoch_msg)
        self.logs.append(epoch_msg)
        self.epoch += 1 if epoch_end else 0
        self.dump_logs()
    
    def dump_logs(self):
        msg = '\n'.join(self.logs)
        with open(self.dest_path, 'w') as f:
            f.write(msg)
