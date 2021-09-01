import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import MiniImageNetDataset
from sampler import TestSampler
from models import PrototypicalNet
from trainer import ProtoNetTrainer
from utils import _worker_init_fn, _ensure_dir, set_random_seed


def main():

    parser = argparse.ArgumentParser(description='Script to test.')
    parser.add_argument('-fc_dim',  default=64,  type=int)
    parser.add_argument('-M',       default=20,  type=int)
    parser.add_argument('-shot',    default=1,  type=int)
    parser.add_argument('-ifi',     default=False, action='store_true')
    parser.add_argument('-hall',    default=False, action='store_true')
    parser.add_argument('-distance_type', default='euclid', type=str)
    
    parser.add_argument('-model_path',   type=str)
    parser.add_argument('-img_dir',      type=str)
    parser.add_argument('-img_csv_path', type=str)
    parser.add_argument('-testcase_csv_path', type=str)
    parser.add_argument('-output_csv_path',  type=str)
    args = parser.parse_args()

    cfg = {   
        "seed": 2021,
        "learning_rate": 0,
        "epochs": 0,
        "train_way": 0,
        "query": 15,

        "val_way": 5, 
        "shot": 1, 
        "fc_dim": args.fc_dim, 
        "distance": "euclid", 
        "M": args.M
    }   

    set_random_seed(cfg['seed'])

    test_dset = MiniImageNetDataset(
        csv_path=args.img_csv_path, 
        img_dir=args.img_dir,
        mode='test'
    )
    
    test_sampler = TestSampler(episode_file_path=args.testcase_csv_path)
    test_loader = DataLoader(test_dset, batch_sampler=test_sampler, pin_memory=True)

    model = PrototypicalNet(
        hid_dim=cfg['fc_dim'], 
        m=cfg['M'] if args.hall else 0,
        ifi=args.ifi,
        distance_type=args.distance_type
    )

    trainer = ProtoNetTrainer(model, cfg)
    trainer.load_ckpt(args.model_path)
    predictions = trainer.meta_test(test_loader)

    with open(args.output_csv_path, "w") as f:
        header = "episode_id," + ",".join([f"query{i}" for i in range(75)]) + "\n"
        f.write(header)

        for episode_id, pred_list in enumerate(predictions):
            pred_str = f"{episode_id}," + ",".join([str(x) for x in pred_list]) + "\n"
            f.write(pred_str)
            


if __name__ == '__main__':
    main()

