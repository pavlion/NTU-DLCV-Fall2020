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


def main(args):
    #print("Args:", args)
    
    # with open(os.path.join("src", "config", "ProtoNet_config.json"), "r") as f:
    #     train_cfg = json.load(f)
    #     print(train_cfg)
    train_cfg = {   
        "seed": 2021,
        "epochs": args.epoch, 
        "learning_rate": args.lr, 
        "train_way": args.train_way, 
        "val_way": args.val_way, 
        "shot": args.shot, 
        "query": args.query, 
        "train_iter": 100, 
        "val_iter": 400,
        "fc_dim": args.fc_dim,
        "distance": args.distance
    }   
    set_random_seed(train_cfg['seed'])

#     train_cfg = ProtoNetConfig(
#         epochs=100, 
#         learning_rate=0.001,
#         train_way=30, 
#         val_way=5, 
#         shot=1,
#         query=15,
#         train_iter=100,
#         val_iter=400
#     )
#     print(train_cfg.get_cfg())
    print("Config:\n", "\n".join([f"{k}: {v}" for k, v in train_cfg.items()]), "\n")
    train_loader, val_loader = get_data_loaders(train_cfg)
    model = PrototypicalNet(hid_dim=train_cfg['fc_dim'], distance_type=train_cfg['distance'])
    
    trainer = ProtoNetTrainer(model, train_cfg)

    print("Start Training!")
    trainer.train([train_loader, val_loader], dest_dir=args.dest_dir)

def get_data_loaders(cfg):
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
    train_sampler = CategoriesSampler(label=train_dset.label, num_batch=cfg['train_iter'], 
                        classes_per_iter=cfg['train_way'], num_samples=cfg['shot']+cfg['query'])
    val_sampler = CategoriesSampler(label=val_dset.label, num_batch=cfg['val_iter'], 
                        classes_per_iter=cfg['val_way'], num_samples=cfg['shot']+cfg['query'])

    train_loader = DataLoader(train_dset, batch_sampler=train_sampler, pin_memory=True,
        num_workers=4)#, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_dset, batch_sampler=val_sampler, pin_memory=True, 
        num_workers=4)#, worker_init_fn=_worker_init_fn)

    return train_loader, val_loader

def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-epoch', '-e', default=100, type=int)
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-seed', default=2020, type=int)
    parser.add_argument('-train_way', default=30, type=int)
    parser.add_argument('-val_way', default=5, type=int)
    parser.add_argument('-shot', default=1, type=int)
    parser.add_argument('-query', default=15, type=int)
    parser.add_argument('-distance', default='euclid', type=str)
    parser.add_argument('-fc_dim',  default=64,  type=str)
    parser.add_argument('-dest_dir',  default=None,  type=str)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)

