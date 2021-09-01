import os

import PIL
import pandas as pd
import os.path as osp

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

filenameToPILImage = lambda x: PIL.Image.open(x).convert('RGB')

class MiniImageNetDataset(Dataset):
    def __init__(self, csv_path, img_dir, mode='test'):
        self.mode = mode
        self.img_dir = img_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.label = self.data_df.loc[:, "label"]
        self.name_to_id = {k: i for i, k in enumerate(set(self.label))}
        
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        file_name = self.data_df.loc[index, "filename"]
        image = self.transform(os.path.join(self.img_dir, file_name))
        if self.mode != 'test':            
            label = self.data_df.loc[index, "label"]
            label = self.name_to_id[label]
            instance = {'img': image, 'label': label, 'file_name': file_name}
        
        else: 
            instance = {'img': image, 'file_name': file_name}
                    
        return instance 

    def __len__(self):
        return len(self.data_df)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from sampler import CategoriesSampler, TestSampler

    # train_dset = MiniImageNetDataset(
    #     csv_path=os.path.join("hw4_data", "train.csv"), 
    #     img_dir=os.path.join("hw4_data", "train") 
    # )

    # train_way = 5
    # shot = 1
    # query = 4
    # train_sampler = CategoriesSampler(label=train_dset.label, num_batch=200, 
    #                     classes_per_iter=train_way, num_samples=shot+query)

    # train_loader = DataLoader(train_dset, pin_memory=False, batch_sampler=train_sampler)

    # for batch in train_loader:
    #     print(batch.keys())
    #     print(type(batch['img']), type(batch['label']))
    #     print(batch['img'].shape)
    #     print(batch['label'])
    #     break

    test_dset = MiniImageNetDataset(
        csv_path=os.path.join("hw4_data", "val.csv"), 
        img_dir=os.path.join("hw4_data", "val"),
        mode='test'
    )
    
    test_sampler = TestSampler(episode_file_path=os.path.join("hw4_data", "val_testcase.csv"))
    test_loader = DataLoader(test_dset, batch_sampler=test_sampler, pin_memory=True)
    for batch in test_loader:
        print(batch['img'].shape)
        continue