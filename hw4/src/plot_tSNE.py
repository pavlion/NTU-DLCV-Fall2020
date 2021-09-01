import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import MiniImageNetDataset
from sampler import TestSampler
from models.PrototypicalNet import ProtoNet_tSNE
from trainer import ProtoNetTrainer
from trainer.ProtoNetTrainer import make_label
from utils import _worker_init_fn, _ensure_dir, set_random_seed

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser(description='Script to test.')
    parser.add_argument('-seed',  default=2021,  type=int)
    parser.add_argument('-fc_dim',  default=64,  type=int)
    parser.add_argument('-M',       default=20,  type=int)
    parser.add_argument('-ifi',     default=False, action='store_true')
    parser.add_argument('-hall',    default=False, action='store_true')    
    parser.add_argument('-model_path',   type=str)
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = 'cuda'
    args.dest_dir = os.path.join(
        "results", 
        "tSNE_hall_ifi.png" if args.ifi else "tSNE_hall.png"
    )

    test_dset = MiniImageNetDataset(
        csv_path=os.path.join("hw4_data", "val.csv"), 
        img_dir=os.path.join("hw4_data", "val"),
        mode='val'
    )
    
    test_sampler = TestSampler(episode_file_path=os.path.join("hw4_data", "val_testcase.csv"))
    test_loader = DataLoader(test_dset, batch_sampler=test_sampler, pin_memory=True)

    model = ProtoNet_tSNE(
        hid_dim=64, 
        m=args.M if args.hall else 0,
        ifi=args.ifi,
        distance_type='euclid'
    ).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    proto_list = torch.FloatTensor([])
    hall_list = torch.FloatTensor([])
    ifi_list = torch.FloatTensor([])
    label_list = torch.LongTensor([])
    with torch.no_grad():
        for i, batch in enumerate(test_loader, start=1):
            
            len_support = 5 * 1 # way * shot
            img = batch['img'].to(device) 
            label = batch['label']
            label = make_label(label[:len_support], label[len_support:], 5) # (query, )

            shot = img[: len_support]  # (5, 3, 84, 84)
            query = img[len_support: ] # (75, 3, 84, 84)
            
            proto_s, i_data, ifi_data = model(shot, query, num_way=5)

            hid_dim = proto_s.size(2)
            proto_s = proto_s.cpu().view(-1, hid_dim) # (shot*way, hid)
            hall = i_data.cpu().view(-1, hid_dim) # (m*way, hid)
            ifi = ifi_data.cpu().view(-1, hid_dim)  # (m*way, hid)

            proto_list = torch.cat((proto_list, proto_s.cpu()), dim=0) # (5, hid)
            hall_list = torch.cat((hall_list, hall), dim=0)   # (5*M, hid)
            ifi_list = torch.cat((ifi_list, ifi), dim=0)
            
            label_list = torch.cat((label_list, label[:5]), dim=0) # (5, hid)

            if i % 50 == 0:
                print(f"Inferencing: {i}/{len(test_loader)}", end="\r")
                # break

    print([x.shape for x in (proto_list, hall_list, ifi_list, label_list)])
    # proto_list: (N, hid) N=num_episode*way
    # hall_list:  (N*(Q/W)*M, hid)  (Q/W: num of query for each class)
    # ifi_list:   (N*(Q/W)*M, hid) 
    # label_list: (N, hid)
    
    # X: (N + N*M + N*M, hid_dim)
    # y: (N + N*M + N*M, )
    N = proto_list.size(0)
    X = torch.cat((proto_list, hall_list, ifi_list), dim=0)   
    y = torch.cat((label_list, label_list.repeat(args.M)), dim=0)
    if args.ifi:
        y = torch.cat((y, label_list.repeat(args.M)), dim=0)

    print(N, [x.shape for x in (X, y)])
    # X, y = proto_list, label_list
    colors = np.random.rand(5, 3)
    colors_hall = np.random.rand(5, 3)
    colors_ifi = np.random.rand(5, 3)
    

    print("Fitting t-SNE" + " "*20)
    tsne = TSNE(n_components=2, init='pca', random_state=args.seed)
    X_tsne = tsne.fit_transform(X)
    X_norm = (X_tsne - X_tsne.min(0)) / (X_tsne.max(0) - X_tsne.min(0))
    
    print("Plotting t-SNE")
    # plt.figure(figsize=(5, 5))
    # for i in range(X_norm.shape[0]):

    #     if i >= N:
    #         marker = '^'
    #         color = colors_hall
    #         #feature_type = 'Hallucinated features'
    #     else:
    #         marker = 'x'
    #         color = colors
    #         #feature_type = 'Real features'

    #     plt.plot(X_norm[i, 0], X_norm[i, 1], 
    #         marker=marker, color=color[y[i]])#, label=feature_type)

    #plt.legend()
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(f"t-SNE for different types of features")
    # plt.savefig(args.dest_dir)
    # plt.close()

    plt.figure(figsize=(12, 12))
    for i in range(5):
        X_part, y_part = X_norm[:N], y[:N]
        plt.scatter(X_part[y_part == i, 0], X_part[y_part == i, 1], marker='x',
                    color=colors[i], s=5, label=f"Real features of class {i+1}")

    for i in range(5):
        X_part, y_part = X_norm[N: N+N*args.M], y[N: N+N*args.M]
        plt.scatter(X_part[y_part == i, 0], X_part[y_part == i, 1], marker='^',
                    color=colors_hall[i], s=5, label=f"Hallucinated features of class {i+1}")
        #print(y, "\n\n")
        #plt.scatter([], [], color=colors[i], label=f"Class{i+1}")

    if args.ifi:
        for i in range(5):
            X_part, y_part = X_norm[N+N*args.M:], y[N+N*args.M:]
            plt.scatter(X_part[y_part == i, 0], X_part[y_part == i, 1], marker='^',
                        color=colors_ifi[i], s=5, label=f"Secondly Hallucinated features of class {i+1}")


    plt.legend(fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"t-SNE for different types of features")
    plt.savefig(
        os.path.join("results", 
        "tSNE_hall_ifi_sep.png" if args.ifi else "tSNE_hall_sep.png")
    )
    plt.close()




if __name__ == '__main__':
    main()

