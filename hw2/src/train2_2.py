import os
import argparse
import torch
torch.manual_seed(422)

from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from models import FCN32s
from callbacks import MetricsLogger
from trainer import P2Trainer


def main(args):
    print("Args:", args)
    
    print("Getting Datasets")
    train_dset = SegmentationDataset(data_path=os.path.join('hw2_data', 'p2_data', 'train'), mode='train')
    val_dset = SegmentationDataset(data_path=os.path.join('hw2_data', 'p2_data', 'validation'), mode='val')

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
    model = FCN32s(num_class=7)

    print("Setting up trainer")
    trainer = P2Trainer(
        max_epochs=args.epoch,
        learning_rate=args.lr,
        optimizer_opt='Adam',
        loss_opt='CELoss',
        device='cuda:1',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    working_dir = os.path.join(args.model_dir, f"{model.__name__}_lr{args.lr}")
    if not os.path.exists(working_dir): 
        os.makedirs(working_dir)
    
    print("Setting up logger")
    metric_logger = MetricsLogger(os.path.join(working_dir, "log.json"))

    print('Start training!')    
    trainer.train(callbacks=[metric_logger], ckpt_path=working_dir)



def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_dir', '-m', default='ckpt/p2/', type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-epoch', '-e', default=30, type=int)
    parser.add_argument('-lr', default=3e-3, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
