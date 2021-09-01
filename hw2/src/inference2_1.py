import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ClassificationDataset
from models import ResNet18, ResNext101
from trainer import P1Trainer

def main(args):
    print("Arguments:", args)
    
    val_dset = ClassificationDataset(
        data_path=args.data_path, 
        mode='test' # label is not given
    )

    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dset.collate_fn
    )

    model = ResNet18(out_dim=50)
    #model = ResNext101(out_dim=50)
    model.load_state_dict(torch.load(args.model_path, map_location="cuda")['model'])
    
    trainer = P1Trainer(model=model)
    #trainer.load(args.model_path)
    
    predictions = trainer.predict(val_loader, no_label=True)

    pred_str = "image_id,label\n"
    for id, pred in zip(predictions['id'], predictions['pred']):
        pred_str += "{},{}\n".format(id, pred)
    
    with open(os.path.join(args.dest_dir, 'test_pred.csv'), 'w') as f:
        f.write(pred_str)

    print('Done.' + ' '*50)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_path', type=str)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-dest_dir', type=str)
    parser.add_argument('-batch_size', default=64, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
