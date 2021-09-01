import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

import torch
torch.manual_seed(422)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataset import ClassificationDataset
from models import ResNet18_tSNE

def main(args):
    print("Arguments:", args)    

    print("Getting Datasets.")
    val_dset = ClassificationDataset(
        data_path=os.path.join('hw2_data', 'p1_data', 'val_50'), 
        mode='val'
    )

    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dset.collate_fn
    )

    print("Setting up model." + " "*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18_tSNE(out_dim=50).to(device)
    model.resnet.load_state_dict(torch.load(args.model_path)['model']) 

    print('Inferencing.')
    model.eval()
    predictions, labels = [], []
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            seq = batch['img'].to(device)
            label = batch['label'].unsqueeze(1).long()
            features = model(seq) 

        for f, l in zip(features, label):
            predictions += [f.cpu().numpy()]
            labels += [l.numpy()]
        
        print("Predicting {}/{}".format(i+1, len(val_loader)), end='\r')
    print('Finish predicting.')


    X = np.concatenate(predictions).reshape(len(predictions), -1)
    y = np.concatenate(labels)
    colors = np.random.rand(50,3)

    print("Fitting t-SNE")
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    
    print("Plotting t-SNE")
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min) 
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(
            X_norm[i, 0], X_norm[i, 1], str(y[i]), 
            color=colors[y[i]],
            fontdict={'weight': 'bold', 'size': 9}
        )
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.dest_path)  



    print('Done.' + ' '*50)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_path', '-m', type=str)
    parser.add_argument('-dest_path', '-d', type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-device', '-dev', default='cuda', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
