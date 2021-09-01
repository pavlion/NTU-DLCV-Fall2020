import os 

class Metric():
    def __init__(self):
        pass
    def reset(self):
        pass
    def update(self): 
        pass

class Accuracy(Metric):
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

class TrainingLogger(Metric):

    def __init__(self, max_epochs, dest_dir='.'):
        self.epoch = 0
        self.logs = []
        self.dest_dir = dest_dir 
    
    def reset(self, dest_dir='.'):
        self.logs = [] 
        self.epoch = 0
        self.dest_dir = dest_dir 

    def print_and_update(self, epoch_msg, epoch_end=False):
        self.logs.append(epoch_msg)
        self.epoch += 1 if epoch_end else 0
        print(epoch_msg)
        self.dump_logs()
    
    def dump_logs(self):
        msg = '\n'.join(self.logs)
        with open(os.path.join(self.dest_dir, 'epoch_logs.txt'), 'w') as f:
            f.write(msg)

