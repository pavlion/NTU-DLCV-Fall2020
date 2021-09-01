import os 
import random
import torch
import numpy as np

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def set_random_seed(seed):
    #print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
