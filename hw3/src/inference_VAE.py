import os
import PIL
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from trainer3_1 import P1Trainer
from callbacks import MetricsLogger
from models import VAE
from dataset import FaceDataset


def main(args):
    print("Args:", args)
    torch.manual_seed(422)
    
    if not args.generate:
        print("Getting Datasets")
        data_path = os.path.join('hw3_data', 'face', 'report')
        val_dset = FaceDataset(
            data_path=os.path.join('hw3_data', 'face', 'report'),
            label_path=None,
            mode='test'
        )
        
        val_loader = DataLoader(
            dataset=val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_dset.collate_fn
        )
    
    print("Setting up model" + " "*50)
    model = VAE(4096)

    print("Setting up trainer")
    trainer = P1Trainer(
        kld_weight=args.kld_weight,
        model=model
    )
    trainer.load(args.model_path)
    
    if args.reconstruct:
        print('Inferencing3.1.3')
        generated_imgs = trainer.predict(val_loader)

        losses = {}
        for recon_img, recon_loss, file_name \
                in zip(generated_imgs['imgs'], generated_imgs['loss'], generated_imgs['file_name']):
            img = PIL.Image.fromarray(np.uint8(recon_img*255.0))
            img.save(os.path.join(args.dest_dir, f'{file_name}_recon.png'))
            losses[file_name] = recon_loss

        with open(os.path.join(args.dest_dir, "losses.json"), "w") as f:
            json.dump(losses, f)

    if args.generate: 
        print('Inferencing3.1.4')
        generated_imgs = trainer.predict(num_samples=args.num_samples)  
        vutils.save_image(
            generated_imgs, 
            #fp=os.path.join(args.dest_dir, f"randomly_generated.png"),
            fp=args.dest_path,
            normalize=True
        )
        # for ii in range(args.num_samples):
            # img = PIL.Image.fromarray(np.uint8(generated_imgs[ii]*255.0))
            # img.save(os.path.join(args.dest_dir, f'{ii}.png'))

    if args.tsne:
        print('Inferencing3.1.5')
        
        val_dset = FaceDataset(
            data_path=os.path.join('hw3_data', 'face', 'test'),
            label_path=os.path.join('hw3_data', 'face', 'test.csv'),
            mode='val'
        )        
        val_loader = DataLoader(
            dataset=val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_dset.collate_fn
        )
        model.eval()
        latent_vecs, labels = [], []
        with torch.no_grad():
            for ii, batch in enumerate(val_loader):
                img = batch['img'].to(trainer.device)
                label = batch['label'] # [B, A] A=# of attribute
                latent_vec = model.get_latent_vec(img).cpu() # [B, H]
                #latent_vec = model.encode_block(img).view(-1, 4096).cpu()
                
                latent_vecs.append(latent_vec)
                labels.append(label)

        latent_vecs = torch.cat(latent_vecs, dim=0).numpy() # [N, H]
        labels = torch.cat(labels, dim=0).int().numpy() # [N, A]

        for i in range(labels.shape[1]):
        #for i in [12]:
            print("Processing with attribute", i)
            labels_ = labels[: , i]
            plot_tSNE(latent_vecs, labels_, num_class=2, dest_path=os.path.join(args.dest_path, f"tSNE_attr{i}.png"))

    print("\n")

def plot_tSNE(X, y, num_class, dest_path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    colors = np.random.rand(num_class,3)

    print("Fitting t-SNE")
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    
    print("Plotting t-SNE")
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # [N, 2]
    plt.figure(figsize=(5, 5))
    # for i in range(X_norm.shape[0]):
        # plt.text(
        #     X_norm[i, 0], X_norm[i, 1], 
        #     str(y[i]), 
        #     color=colors[y[i]],
        #     fontdict={'weight': 'bold', 'size': 9}
        # )
    
    for i in range(colors.shape[0]):
        plt.scatter(X_norm[y == i, 0], X_norm[y == i, 1], c=colors[i].reshape(1, -1), s=5, label=f"class {i}")
    
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(dest_path)  

    print('Done.' + ' '*10)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_path', '-p', default="ckpt", type=str)
    parser.add_argument('-dest_path', '-d', default="results/VAE.png", type=str)
    parser.add_argument('-batch_size', '-b', default=32, type=int)
    parser.add_argument('-kld_weight', default=1e-4, type=float)
    parser.add_argument('-num_samples', default=32, type=int)
    parser.add_argument('--reconstruct', '-r', action='store_true', default=False)
    parser.add_argument('--generate', '-g', action='store_true', default=False)
    parser.add_argument('--tsne', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
