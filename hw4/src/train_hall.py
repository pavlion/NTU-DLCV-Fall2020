import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import MiniImageNetDataset
from sampler import CategoriesSampler
from models import PrototypicalNet
from trainer import ProtoNetTrainer, ProtoNetConfig
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
    parser.add_argument('-fc_dim',  default=64,  type=int)
    parser.add_argument('-M',       default=20, type=int)
    parser.add_argument('-dest_dir',  default=None,  type=str)
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

    train_loader, val_loader = get_data_loaders(train_cfg)
    model = PrototypicalNet(
        hid_dim=train_cfg['fc_dim'], 
        m=train_cfg['M'],
        distance_type='euclid'
    )
    
    trainer = ProtoNetTrainer(model, train_cfg)
    trainer.cfg_str = f"{train_cfg['val_way']}way{train_cfg['shot']}shot_fc{train_cfg['fc_dim']}_M{train_cfg['M']}"

    trainer.train([train_loader, val_loader], dest_dir=args.dest_dir)



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

if __name__ == '__main__':
    main()

