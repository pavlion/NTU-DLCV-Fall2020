import os
import argparse
import numpy as np
import scipy
import PIL
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from models import FCN8s, FCN32s
from callbacks import MetricsLogger
from trainer import P2Trainer



def main(args):
    print("Arguments:", args)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    

    print("Getting Datasets.")
    val_dset = SegmentationDataset(
        data_path=args.data_path, 
        mode='test'
    )

    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dset.collate_fn
    )

    print("Setting up model." + " "*50)
    model = FCN8s(num_class=7) if args.improved else FCN32s(num_class=7)


    print("Setting up trainer.")
    trainer = P2Trainer(model=model)
    trainer.load(args.model_path)
    

    print('Inferencing.')
    predictions = trainer.predict(
        val_loader, 
        dest_dir=args.dest_dir,
        no_label=True
    )   

    class_color = {
        0:  [0, 255, 255],
        1:  [255, 255, 0],
        2:  [255, 0, 255],
        3:  [0, 255, 0],
        4:  [0, 0, 255],
        5:  [255, 255, 255],
        6:  [0, 0, 0],
    }
    
    ids = np.array(predictions['id'])
    preds = np.array(predictions['pred'])
    H, W = preds[0].shape
    #print(ids.shape, preds.shape)
    
    # Visualize results
    for ii, (id, pred) in enumerate(zip(ids, preds)): # pred = [H, W]
        mask_img = np.zeros((H, W, 3))
        
        # change 0~6 to their corresponding color
        for cls, val in class_color.items(): 
            mask_img[pred==cls] = np.array(val)
        
        img = PIL.Image.fromarray(np.uint8(mask_img))
        #img = img.resize((512, 512), resample=PIL.Image.BICUBIC) #PIL.Image.LANCZOS
        img.save(os.path.join(args.dest_dir, f'{id}_mask.png'))
        
        print("Dumping masks {}/{}".format(ii+1, len(predictions['id'])), end='\r')

    print('Done.' + ' '*50)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-model_path', '-m', type=str)
    parser.add_argument('-dest_dir', '-d', type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('--improved', '-i', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
