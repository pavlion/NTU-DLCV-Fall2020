import os
import argparse
import torch
torch.manual_seed(422)
from torch.utils.data import DataLoader

from models import DCGAN, DCGAN_Config
from dataset import FaceDataset


def main(args):
    print("Args:", args)

    print("Getting Datasets")
    data_path = os.path.join('hw3_data', 'face', 'train')
    label_path = os.path.join('hw3_data', 'face', 'train.csv')
    train_dset = FaceDataset(data_path, label_path, mode='train')

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dset.collate_fn
    )

    print("Setting up model" + " "*50)
    cfg = DCGAN_Config(
        batch_size=args.batch_size,
        latent_size=args.latent_size,    
        epochs=args.epoch,
        learning_rate=args.lr
    )
    print("Model Config:", cfg.get_cfg_dict())
    model = DCGAN(cfg)

    print("Start Training!")
    working_dir = os.path.join(args.ckpt_dir, f"GAN_latent{args.latent_size}")
    _check_dir(working_dir)
    _check_dir(os.path.join(working_dir, "gen_img"))
    model.train(train_loader, dest_dir=working_dir)
    
    
def _check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-ckpt_dir', '-p', default="ckpt", type=str)
    
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-latent_size', '-l', default=100, type=int)
    parser.add_argument('-epoch', '-e', default=30, type=int)
    parser.add_argument('-lr', default=2e-3, type=float)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
