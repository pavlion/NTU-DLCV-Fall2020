import os
import argparse
import torch
torch.manual_seed(422)
from torch.utils.data import DataLoader
from trainer3_1 import P1Trainer
from callbacks import MetricsLogger
from models import VAE
from dataset import FaceDataset


def main(args):
    print("Args:", args)

    print("Getting Datasets")
    train_dset = FaceDataset(
        data_path=os.path.join('hw3_data', 'face', 'train'), 
        label_path=os.path.join('hw3_data', 'face', 'train.csv'), 
        mode='train'
    )
    val_dset = FaceDataset(
        data_path=os.path.join('hw3_data', 'face', 'test'), 
        label_path=os.path.join('hw3_data', 'face', 'test.csv'), 
        mode='val'
    )

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dset.collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dset.collate_fn
    )
    print("Setting up model" + " "*50)
    model = VAE(args.latent_size)

    working_dir = os.path.join(args.model_dir, f"VAE{args.latent_size}_lr{args.lr}_kldw_{args.kld_weight}")
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    print("Setting up trainer")
    trainer = P1Trainer(
        max_epochs=args.epoch,
        learning_rate=args.lr,
        kld_weight=args.kld_weight,
        optimizer_opt='Adam',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    trainer.load("results/VAE2048_lr0.001_kldw_0.0001/best_loss_epoch100_MSE0.0236_KLD106.1755.pth")

    print("Setting up logger")
    metric_logger = MetricsLogger(os.path.join(working_dir, "log.json"))

    print('Start training!')
    trainer.train(callbacks=[metric_logger], ckpt_path=working_dir)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_dir', '-p', default="ckpt", type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-epoch', '-e', default=30, type=int)
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-latent_size', default=2048, type=int)
    parser.add_argument('-kld_weight', default=1e-4, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
