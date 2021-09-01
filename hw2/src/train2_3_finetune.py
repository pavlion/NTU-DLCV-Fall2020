from trainer import P2Trainer
from callbacks import MetricsLogger
from models import *
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import os
import argparse
import torch
torch.manual_seed(422)


def main(args):
    print("Args:", args)

    model_name = args.model_name.lower()
    assert model_name in ('fcn32s', 'fcn16s', 'fcn8s', 'fcn4s', 'deeplab',
                          'fcn2s', 'segnet', 'unet'), 'invalid model name'

    print("Getting Datasets")
    train_dset = SegmentationDataset(data_path=os.path.join(
        'hw2_data', 'p2_data', 'train'), mode='train')
    val_dset = SegmentationDataset(data_path=os.path.join(
        'hw2_data', 'p2_data', 'validation'), mode='val')

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
    if model_name == 'fcn2s':
        model = FCN2s(num_class=7)
    elif model_name == 'fcn4s':
        model = FCN4s(num_class=7)
    elif model_name == 'fcn8s':
        model = FCN8s(num_class=7)
    elif model_name == 'fcn16s':
        model = FCN16s(num_class=7)
    elif model_name == 'fcn32s':
        model = FCN32s(num_class=7)
    elif model_name == 'segnet':
        model = SegNet(num_class=7)
    elif model_name == 'unet':
        model = UNet16(num_classes=7, is_deconv=True)
    elif model_name == 'deeplab':
        model = DeepLabv3_ResNet101(num_classes=7)

    working_dir = os.path.join(args.model_dir, f"{model.__name__}_lr{args.lr}")
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        
    ckpt = torch.load("ckpt/p2/DeepLabv3_ResNet101_lr5e-05/DeepLabv3_ResNet101_loss0.346967_miou0.728019.ckpt")
    model.load_state_dict(ckpt['model'])

    print("Setting up trainer")
    trainer = P2Trainer(
        max_epochs=args.epoch,
        learning_rate=args.lr,
        optimizer_opt='Adam',
        #loss_opt='CELoss',
        loss_opt='Lovasz',
        # scheduler_opt='StepLR',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

    print("Setting up logger")
    metric_logger = MetricsLogger(os.path.join(working_dir, "log.json"))

    print('Start training!')
    trainer.train(callbacks=[metric_logger], ckpt_path=working_dir)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_dir', '-m', default='ckpt/p2/', type=str)
    parser.add_argument('-model_name', '-n', default='FCN8s', type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-epoch', '-e', default=30, type=int)
    parser.add_argument('-lr', default=7e-5, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
