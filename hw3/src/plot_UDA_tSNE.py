import os
import argparse
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataset import DigitDataset
from models import DANN
from models.SWD import SWD


def main(args):
    print("Arguments:", args)
    torch.manual_seed(422)
    args.dest_path = os.path.join(args.dest_path)

    val_loaders = {
        'mnistm': get_loader('mnistm', args),
        'usps': get_loader('usps', args),
        'svhn': get_loader('svhn', args)
    }

    # model_paths = {
    #     'DANN': {
    #         'mnistm-svhn': os.path.join("ckpt", "DANN", "DANN_mnistm-svhn.pth"),
    #         'svhn-usps':   os.path.join("ckpt", "DANN", "DANN_svhn-usps.pth"),
    #         'usps-mnistm': os.path.join("ckpt", "DANN", "DANN_usps-mnistm.pth")
    #     },
    #     'SWD': {
    #         'mnistm-svhn': os.path.join("ckpt", "SWD", 'mnistm2svhn', "best_loss_G_49.5776.pth"),
    #         'svhn-usps':   os.path.join("ckpt", "SWD", 'svhn2usps', "best_acc_G_72.1393.pth"),
    #         'usps-mnistm': os.path.join("ckpt", "SWD", 'usps2mnistm', "best_acc_G_50.2_k=10.pth")
    #     }
    # }

    # print("Setting up model." + " "*10)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DANN() if args.model_type == 'DANN' else SWD.Generator()
    # model = model.to(device)
    # # T-SNE for Digits
    # for scenario in ['mnistm-svhn', 'svhn-usps', 'usps-mnistm']:
    #     src_domain, trg_domain = scenario.split('-')
    #     val_dset = ConcatDataset(
    #         (val_loaders[src_domain], val_loaders[trg_domain]))
    #     test_loader = DataLoader(
    #         dataset=val_dset,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         collate_fn=val_loaders[src_domain].collate_fn
    #     )

    #     model = DANN() if args.model_type == 'DANN' else SWD.Generator()
    #     if args.model_type == 'SWD' and src_domain == 'usps':
    #         model = SWD.USPS.Generator()
    #     print(model.load_state_dict(torch.load(model_paths[args.model_type][scenario])))
        
    #     model = model.feature_extractor if args.model_type == 'DANN' else model
            
    #     print(f"Processing case {scenario} for digits")
    #     plot_tSNE_digits(model, test_loader, domains=[src_domain, trg_domain], 
    #                      dest_path=os.path.join(args.dest_path, f"{scenario}_digits.png"))
    #     print(f"Finish processing case {scenario} for digits\n")

    # # t-SNE for domain:
    # for scenario in ['mnistm-svhn', 'svhn-usps', 'usps-mnistm']:
    #     src_domain, trg_domain = scenario.split('-')
    #     src_test_loader = DataLoader(
    #         dataset=val_loaders[src_domain],
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         collate_fn=val_loaders[src_domain].collate_fn
    #     )
    #     trg_test_loader = DataLoader(
    #         dataset=val_loaders[trg_domain],
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         collate_fn=val_loaders[trg_domain].collate_fn
    #     )
        
    #     test_loaders = [src_test_loader, trg_test_loader]
    #     domains = [src_domain, trg_domain]

    #     model = DANN() if args.model_type == 'DANN' else SWD.Generator()
    #     if args.model_type == 'SWD' and src_domain == 'usps':
    #         model = SWD.USPS.Generator()
    #     print(model.load_state_dict(torch.load(model_paths[args.model_type][scenario])))
        
    #     model = model.feature_extractor if args.model_type == 'DANN' else model


    #     print(f"Processing case {scenario} for domains")
    #     plot_tSNE_domains(model, test_loaders, domains, 
    #                      dest_path=os.path.join(args.dest_path, f"{scenario}_domains.png"))
    #     print(f"Finish processing case {scenario} for domains\n")

    # # t-SNE for domain using raw images :
    # for scenario in ['mnistm-svhn', 'svhn-usps', 'usps-mnistm']:
    #     src_domain, trg_domain = scenario.split('-')
    #     src_test_loader = DataLoader(
    #         dataset=val_loaders[src_domain],
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         collate_fn=val_loaders[src_domain].collate_fn
    #     )
    #     trg_test_loader = DataLoader(
    #         dataset=val_loaders[trg_domain],
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         collate_fn=val_loaders[trg_domain].collate_fn
    #     )
        
    #     test_loaders = [src_test_loader, trg_test_loader]
    #     domains = [src_domain, trg_domain]

    #     print(f"Processing case {scenario} for domains")
    #     plot_tSNE_domains_no_model(model, test_loaders, domains, 
    #                      dest_path=os.path.join(args.dest_path, f"{scenario}_domains_raw.png"))
    #     print("\n")

    # t-SNE for digits using raw images :
    for scenario in ['mnistm-svhn', 'svhn-usps', 'usps-mnistm']:
        src_domain, trg_domain = scenario.split('-')
        src_test_loader = DataLoader(
            dataset=val_loaders[src_domain],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_loaders[src_domain].collate_fn
        )
        trg_test_loader = DataLoader(
            dataset=val_loaders[trg_domain],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_loaders[trg_domain].collate_fn
        )
        
        test_loaders = [src_test_loader, trg_test_loader]
        domains = [src_domain, trg_domain]

        print(f"Processing case {scenario} for digits")
        plot_tSNE_digits_no_model(None, test_loaders, domains, 
                         dest_path=os.path.join(args.dest_path, f"{scenario}_digits_raw.png"))
        print("\n")


def get_loader(domain, args):
    val_dset = DigitDataset(
        data_path=os.path.join("..", "hw3_data", "digits", domain, "test"),
        label_path=os.path.join("..", "hw3_data", "digits", domain, "test.csv"),
        mode='val'
    )

    # test_loader = DataLoader(
    #     dataset=val_dset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=val_dset.collate_fn
    # )
    return val_dset #test_loader


def plot_tSNE_digits(model, val_loader, domains, dest_path):
    ''' Evaluate on target `val_loader` with model at `model_path` '''

    device = next(model.parameters()).device

    print('Inferencing.')
    model.eval()
    features, labels = [], []
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            img = batch['image'].to(device)
            label = batch['label'].long()
            feature = model(img).cpu()  # [bs, 3200]

        features.append(feature)
        labels.append(label)
        print("Predicting {}/{}".format(i+1, len(val_loader)), end='\r')
        
    X = torch.cat(features, dim=0)   # features = [N, 3200]
    y = torch.cat(labels, dim=0)     # labels = [N, 1] (digits)
    
    colors = np.random.rand(10, 3)
    print("Fitting t-SNE" + " "*10)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    X_norm = (X_tsne - X_tsne.min(0)) / (X_tsne.max(0) - X_tsne.min(0))
    
    plt.figure(figsize=(5, 5))
    for i in range(colors.shape[0]):
        plt.scatter(X_norm[y == i, 0], X_norm[y == i, 1],
                    s=5, label=f"Digit {i}") #c=colors[i].reshape(1, -1), 

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f"t-SNE for {domains[0]}-{domains[1]} case versus digits")
    plt.savefig(dest_path)
    plt.close()

def plot_tSNE_domains(model, val_loaders, domains, dest_path):
    ''' Evaluate on target `val_loaders` with model at `model_path` '''

    device = next(model.parameters()).device

    print('Inferencing.')
    model.eval()
    features, labels = [], []
    for domain_idx, val_loader in enumerate(val_loaders):
        for i, batch in enumerate(val_loader):
            with torch.no_grad():
                img = batch['image'].to(device)
                label = torch.full(size=(img.size(0), ), fill_value=domain_idx, dtype=torch.long)
                feature = model(img).cpu()  # [bs, 3200]

            features.append(feature)
            labels.append(label)
            print("Predicting {}/{}".format(i+1, len(val_loader)), end='\r')
        
    X = torch.cat(features, dim=0)   # features = [N, 3200]
    y = torch.cat(labels, dim=0)     # labels = [N, 1] (digits)
    
    colors = np.random.rand(2, 3)
    print("Fitting t-SNE" + " "*10)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    X_norm = (X_tsne - X_tsne.min(0)) / (X_tsne.max(0) - X_tsne.min(0))
    
    plt.figure(figsize=(5, 5))
    for i in range(colors.shape[0]):
        plt.scatter(X_norm[y == i, 0], X_norm[y == i, 1],
                    s=5, label=f"{domains[i]}") # c=colors[i].reshape(1, -1), 

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f"t-SNE for {domains[0]}-{domains[1]} case versus domains")
    plt.savefig(dest_path)
    plt.close()

def plot_tSNE_digits_no_model(model, val_loaders, domains, dest_path):
    ''' Evaluate on target `val_loader` with model at `model_path` '''

    print('Inferencing.')
    features, labels = [], []
    for val_loader in val_loaders:
        for i, batch in enumerate(val_loader):
            img = batch['image']
            label = batch['label']
            features.append(img.view(img.size(0), -1))
            labels.append(label)
            print("Predicting {}/{}".format(i+1, len(val_loader)), end='\r')
        
    X = torch.cat(features, dim=0)   # features = [N, 3200]
    y = torch.cat(labels, dim=0)     # labels = [N, 1] (digits)
    
    colors = np.random.rand(10, 3)
    print("Fitting t-SNE" + " "*10)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    X_norm = (X_tsne - X_tsne.min(0)) / (X_tsne.max(0) - X_tsne.min(0))
    
    plt.figure(figsize=(5, 5))
    for i in range(colors.shape[0]):
        plt.scatter(X_norm[y == i, 0], X_norm[y == i, 1],
                    s=5, label=f"Digit {i}") #c=colors[i].reshape(1, -1), 

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f"t-SNE for {domains[0]}-{domains[1]} case versus digits")
    plt.savefig(dest_path)
    plt.close()

def plot_tSNE_domains_no_model(model, val_loaders, domains, dest_path):
    ''' Evaluate on target `val_loaders` with model at `model_path` '''

    print('Inferencing.')
    features, labels = [], []
    for domain_idx, val_loader in enumerate(val_loaders):
        for i, batch in enumerate(val_loader):
            img = batch['image']
            label = torch.full(size=(img.size(0), ), fill_value=domain_idx, dtype=torch.long)

            features.append(img.view(img.size(0), -1))
            labels.append(label)
            print("Predicting {}/{}".format(i+1, len(val_loader)), end='\r')
        
    X = torch.cat(features, dim=0)   # features = [N, 3200]
    y = torch.cat(labels, dim=0)     # labels = [N, 1] (digits)
    
    colors = np.random.rand(2, 3)
    print("Fitting t-SNE" + " "*10)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    X_norm = (X_tsne - X_tsne.min(0)) / (X_tsne.max(0) - X_tsne.min(0))
    
    plt.figure(figsize=(5, 5))
    for i in range(colors.shape[0]):
        plt.scatter(X_norm[y == i, 0], X_norm[y == i, 1],
                    c=None, s=5, label=f"{domains[i]}") # c=colors[i].reshape(1, -1)

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f"t-SNE for {domains[0]}-{domains[1]} case versus domains")
    plt.savefig(dest_path)
    plt.close()


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('-model_type', '-t', type=str, default="DANN")
    parser.add_argument('-dest_path', '-d', type=str, default="results")
    parser.add_argument('-batch_size', '-b', default=128, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)
