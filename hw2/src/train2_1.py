import os
import sys
import json
import pickle
import logging
import argparse
from sklearn.model_selection import KFold

import torch

torch.manual_seed(877)

from torch.utils.data import DataLoader, Subset

from dataset import ClassificationDataset
from models import *
from callbacks import MetricsLogger
from trainer import P1Trainer


def main(args):
    print(args)
    logging.info("Getting Datasets")
    train_dset = ClassificationDataset(data_path=os.path.join('hw2_data', 'p1_data', 'train_50'), mode='train')
    val_dset = ClassificationDataset(data_path=os.path.join('hw2_data', 'p1_data', 'val_50'), mode='val')

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
    logging.info("Setting up model")
    #model = ResNet18(out_dim=50)
    model = ResNet34(out_dim=50)
    #model = ResNext101(out_dim=50)

    logging.info("Setting up trainer")
    trainer = P1Trainer(
        max_epochs=args.epoch,
        learning_rate=args.lr,
        optimizer_opt='Adam',
        loss_opt='CELoss',
        model=model,
        device='cuda:1',
        train_loader=train_loader,
        val_loader=val_loader
    )

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logging.info("Setting up logger")
    metric_logger = MetricsLogger(os.path.join(args.model_dir, "log.json"))

    logging.info('start training!')
    trainer.train(callbacks=[], ckpt_path=args.model_dir)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_dir', '-m', default='ckpt/p1', type=str)
    parser.add_argument('-batch_size', '-b', default=16, type=int)
    parser.add_argument('-epoch', '-e', default=10, type=int)
    parser.add_argument('-lr', default=5e-6, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
