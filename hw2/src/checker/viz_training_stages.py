import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from models import *
from callbacks import MetricsLogger
from trainer import P2Trainer



def main(args):
    print("Arguments:", args)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    

    print("Getting Datasets.")
    val_dset = SegmentationDataset(
        data_path=os.path.join('hw2_data', 'p2_data', 'report_img'), 
        mode='val'
    )

    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dset.collate_fn
    )

    print("Setting up model." + " "*50)

    if args.model_name == 'FCN32s':
        model = FCN32s(num_class=7)
        model_path = {
            'early': 'FCN32s_epoch1_loss0.786453_miou0.309301.ckpt',
            'mid': 'FCN32s_epoch3_loss0.505159_miou0.593229.ckpt',
            'final': 'best_loss_FCN32s_loss0.370613_miou68.513456.ckpt'
        }
    elif args.model_name == 'FCN16s':
        model = FCN16s(num_class=7)
        model_path = {
            'early': 'FCN16s_lr7e-5_loss0.345898_miou68.630065.ckpt',
            'mid': 'FCN16s_lr7e-5_loss0.345898_miou68.630065.ckpt',
            'final': 'FCN16s_lr7e-5_loss0.345898_miou68.630065.ckpt'
        }
    elif args.model_name == 'FCN8s':
        model = FCN8s(num_class=7)
        model_path = {
            'early': 'FCN8s_epoch0_loss0.922119_miou0.238323.ckpt',
            'mid': 'FCN8s_epoch5_loss0.544914_miou0.543206.ckpt',
            'final': 'FCN8s_loss0.34034613966941835_miou0.6972654421002388.ckpt'
        }
    elif args.model_name == 'FCN4s':
        model = FCN4s(num_class=7)
        model_path = {
            'early': 'FCN4s_epoch0_loss1.095041_miou0.173514.ckpt',
            'mid': 'FCN4s_epoch5_loss0.618299_miou0.496410.ckpt',
            'final': 'FCN4s_loss0.410535_miou66.798324.ckpt'
        }
    elif args.model_name == 'FCN2s':
        model = FCN2s(num_class=7)
        model_path = {
            'early': 'FCN2s_epoch3_loss0.769744_miou0.336089.ckpt',
            'mid': 'FCN2s_epoch5_loss0.669889_miou0.403999.ckpt',
            'final': 'FCN2s_loss0.392052_miou0.676885.ckpt'
        }

    print("Setting up trainer.")
    trainer = P2Trainer(
        loss_opt='CELoss',
        device=args.device,
        model=model
    )

    
    for stage in ['early', 'mid', 'final']:
        load_and_predict(
            trainer, 
            val_loader, 
            os.path.join(args.model_path, model_path[stage]), 
            os.path.join(args.dest_dir, stage)
        )

        targer_dir = os.path.join(args.dest_dir, stage)
        for filename in os.listdir(targer_dir):
            os.rename(
                os.path.join(args.dest_dir, stage, filename),
                os.path.join(args.dest_dir, args.model_name+'_'+filename[:-4]+f'_{stage}'+filename[-4:])
            )


    print('Done.' + ' '*50)

def load_and_predict(trainer, val_loader, model_path, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    trainer.load(model_path)
    trainer.predict(val_loader, dest_dir=dest_dir)

def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_path', '-m', type=str)
    parser.add_argument('-model_name', '-n', type=str)
    parser.add_argument('-dest_dir', '-d', type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-device', '-dev', default='cuda', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
