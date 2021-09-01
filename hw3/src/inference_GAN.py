import os
import PIL
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from models import Generator
from dataset import FaceDataset


def main(args):
    print("Args:", args)
    torch.manual_seed(422)
    
    
    print("Setting up model" + " "*50)
    model = Generator(500)
    model.load_state_dict(torch.load(args.model_path))

    print('Inferencing3.2.2')
    z = torch.randn(size=(args.num_samples, args.latent_size, 1, 1))
    generated_imgs = model(z)
        
    vutils.save_image(
        generated_imgs, 
        #fp=os.path.join(args.dest_dir, "GAN_randomly_generated.png"),
        fp=os.path.join(args.dest_path),
        normalize=True,
        #range=(-1, 1)
    )
    print("\n")

def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_path', '-p', default="ckpt", type=str)
    parser.add_argument('-dest_path', '-d', default="results", type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-num_samples', default=32, type=int)
    parser.add_argument('-latent_size', default=500, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
