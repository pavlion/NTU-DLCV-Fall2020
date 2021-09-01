import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import MiniImageNetDataset
from sampler import CategoriesSampler
from models import DTN
from trainer import DTNTrainer
from utils import _worker_init_fn, _ensure_dir, set_random_seed


def main():

    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-epoch',   default=100, type=int)
    parser.add_argument('-lr',      default=1e-3, type=float)
    parser.add_argument('-seed',    default=2021, type=int)
    parser.add_argument('-train_way',default=30, type=int)
    parser.add_argument('-val_way', default=5, type=int)
    parser.add_argument('-shot',    default=1, type=int)
    parser.add_argument('-query',   default=15, type=int)
    parser.add_argument('-fc_dim',  default=1024,  type=int)
    parser.add_argument('-M',       default=20, type=int)
    args = parser.parse_args()

    train_cfg = {   
        "seed": args.seed,
        "epochs": args.epoch, 
        "learning_rate": args.lr, 
        "train_way": args.train_way, 
        "val_way": args.val_way, 
        "shot": args.shot, 
        "query": args.query, 
        "fc_dim": args.fc_dim, 
        "M": args.M
    }   

    set_random_seed(train_cfg['seed'])
    print("Config:\n ", "\n  ".join([f"{k}: {v}" for k, v in train_cfg.items()]), "\n")

    train_iter, val_iter = 100, 400
    train_loader, val_loader = get_data_loaders(train_cfg, train_iter, val_iter)
    
    # Make sure train_iter+val_iter < # of images per class
    # Otherwise, modified the code
    ref_loader = get_ref_loader(train_cfg, ref_iter=train_iter+val_iter)


    model = DTN(
        hid_dim=train_cfg['fc_dim'],
        hall=False, 
        distance_type='cosine'
    )

    # Load weights trained in the first stage
    # ckpt = torch.load(os.path.join("ckpt", "pb3.1", "best_loss_1.06299_5way10shot_euclid.pth"))
    # own_state = model.state_dict()
    # for name, param in ckpt.items():
    #     if name not in own_state:
    #             continue
    #     if isinstance(param, torch.nn.Parameter):
    #         param = param.data
    #     own_state[name].copy_(param)


    # trainer = DTNTrainer(model, train_cfg)
    # trainer.cfg_str = f"{train_cfg['val_way']}way{train_cfg['shot']}shot_nohall_fc{train_cfg['fc_dim']}_M{train_cfg['M']}"

    # trainer.train([train_loader, val_loader, ref_loader], dest_dir=os.path.join("ckpt", "pb3.1"))


    ### Second-stage training ###
    model = DTN(
        hid_dim=train_cfg['fc_dim'],
        hall=True, 
        distance_type='cosine'
    )
    trainer = DTNTrainer(model, train_cfg)

    model_path = os.path.join("ckpt", "pb3.1", f"best_acc_44.780_5way1shot_fc1024_M20.pth")
    model.load_state_dict(torch.load(model_path))
    trainer.train([train_loader, val_loader, ref_loader], dest_dir=os.path.join("ckpt", "pb3.1"))




def get_data_loaders(cfg, train_iter=100, val_iter=400):
    train_dset = MiniImageNetDataset(
        csv_path=os.path.join("hw4_data", "train.csv"), 
        img_dir=os.path.join("hw4_data", "train"),
        mode='train' 
    )
    val_dset = MiniImageNetDataset(
        csv_path=os.path.join("hw4_data", "val.csv"), 
        img_dir=os.path.join("hw4_data", "val"),
        mode='val'
    )
    train_sampler = CategoriesSampler(label=train_dset.label, num_batch=train_iter, 
                        classes_per_iter=cfg['train_way'], num_samples=cfg['shot']+cfg['query'])
    val_sampler = CategoriesSampler(label=val_dset.label, num_batch=val_iter, 
                        classes_per_iter=cfg['val_way'], num_samples=cfg['shot']+cfg['query'])

    train_loader = DataLoader(train_dset, batch_sampler=train_sampler,
        num_workers=4, worker_init_fn=_worker_init_fn, pin_memory=True)
    val_loader = DataLoader(val_dset, batch_sampler=val_sampler, 
        num_workers=4, worker_init_fn=_worker_init_fn, pin_memory=True)

    return train_loader, val_loader

def get_ref_loader(cfg, ref_iter=400):
    ref_dset = MiniImageNetDataset(
        csv_path=os.path.join("hw4_data", "train.csv"), 
        img_dir=os.path.join("hw4_data", "train"),
        mode='train'
    )
    ref_sampler = CategoriesSampler(label=ref_dset.label, num_batch=ref_iter, 
                        classes_per_iter=1, num_samples=cfg['M']*2)

    ref_loader = DataLoader(ref_dset, batch_sampler=ref_sampler, 
        num_workers=4, worker_init_fn=_worker_init_fn, pin_memory=True)

    return ref_loader

if __name__ == '__main__':
    main()

