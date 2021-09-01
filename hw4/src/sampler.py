import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data.sampler import Sampler

class TestSampler(Sampler):
    def __init__(self, episode_file_path):
        self.episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.len_ = len(self.episode_df)

    def __iter__(self):
        for i in range(self.len_):
            seq = self.episode_df.loc[i].values.flatten().tolist()
            yield seq

    def __len__(self):
        return self.len_

class CategoriesSampler(Sampler):
    
    def __init__(self, label, num_batch, classes_per_iter, num_samples):
        '''
        Initialize the Sampler.
        Args:
        - labels: an iterable containing all the labels
        - num_batch: number of iterations (episodes) per epoch
        - classes_per_iter: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        '''
        self.num_batch = num_batch
        self.classes_per_iter = classes_per_iter
        self.num_samples = num_samples

        # store the index for each class
        # key=class name, value=idx of that class
        # Note: numpy is able to handle string type
        label = np.array(label) 
        self.unique_class = list(set(label)) 
        self.num_class = len(self.unique_class)
        self.class_indices = {}
        for c in self.unique_class: # c is a string (class name)
            indices = np.argwhere(label == c).reshape(-1)
            self.class_indices[c] = torch.from_numpy(indices)
        
        self.counter = 0

    def __len__(self):
        return self.num_batch
    
    def __iter__(self):
        self.counter += 1
        
        for i in range(self.num_batch):
            # refresh seed
            torch_states = fork_rng_state()
            #torch.manual_seed(i)  #### FAIL!!!
            torch.manual_seed((self.counter << 8) + (i << 12)) # "+" is higher than "<<"
            #torch.manual_seed(time.time()+i) # also: torch.seed()+i
            batch = []
            # classes: selected classes in that batch
            # pos: selected samples in that batch
            class_idx = torch.randperm(self.num_class)[:self.classes_per_iter]
            #print(class_idx)
            for idx in class_idx:
                class_name = self.unique_class[idx]
                idx = self.class_indices[class_name] # indices of class_name
                pos = torch.randperm(len(idx))[:self.num_samples] # pick first num_samples
                batch.append(idx[pos])
            batch = torch.stack(batch).t().reshape(-1).tolist()           
            
            # restore seed for later training
            restore_rng_state(*torch_states)
            yield batch

        # Remarks on "transposing" in yielding batch:
        # As collected indices have classes in the format: [[c1, c1,...], [c2, c2,...], ...]
        # we want each batch to contain different classes: [[c1, c2, ...], [c1, c2, ...], ...]
        # Hence the transpose operation is needed

        # Remarks on not using torch.manual_seed(i):
        # seeding `i` will lead to identical results 
        # everytime it invoke the sampler (invoke once a batch)
        # That is, sampled id is the same for every batch, 
        # which results in poor results.
        # Hence, we set different seed every time sampler is involed 
        # as we want variation between different batches
        # Here, I use a counter to seed so that the reproducibility is maintained.


def fork_rng_state():
    num_devices = torch.cuda.device_count()
    devices = list(range(num_devices))

    cpu_rng_state = torch.get_rng_state()
    gpu_rng_states = []
    for device in devices:
        gpu_rng_states.append(torch.cuda.get_rng_state(device))

    return cpu_rng_state, gpu_rng_states

def restore_rng_state(cpu_rng_state, gpu_rng_states):
    num_devices = torch.cuda.device_count()
    devices = list(range(num_devices))

    torch.set_rng_state(cpu_rng_state)
    for device, gpu_rng_state in zip(devices, gpu_rng_states):
        torch.cuda.set_rng_state(gpu_rng_state, device)

if __name__ == "__main__":
    cfg = {
        'train_way':10, 
        'val_way':5, 
        'shot':2,
        'query':3
    }
    labels = torch.cat([torch.randperm(20) for _ in range(20)])
    sampler = CategoriesSampler(labels, num_batch=5, 
            classes_per_iter=cfg['train_way'], num_samples=cfg['shot']+cfg['query'])
    #ind = next(iter(sampler))
    print("Config:", cfg)
    # print(labels[next(iter(sampler))])
    
    for i in range(20):
        for ind in sampler:
            continue

    