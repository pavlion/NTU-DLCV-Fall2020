import os
import torch
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DigitDataset(Dataset):
    VALID_DOMAIN = ('usps', 'svhn', 'mnistm')

    def __init__(self, data_path, label_path, mode='test', image_size=28, gray2rgb=False):
        assert mode in ('train', 'val', 'test'), f"mode:{mode} is invalid."
        # assert domain in DigitDataset.VALID_DOMAIN, f"source domain:{domain} is invalid."

        self.mode = mode
        self.gray2rgb = gray2rgb
        self.data_path = data_path
        self.label_path = label_path if mode != 'test' else None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # input should have shape [H, W, 3]
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.5, 0.5, 0.5],
            #     std=[0.5, 0.5, 0.5]
            # )
        ])

        # [img, label, file_id]
        self.data = self._parse_data(data_path, label_path, mode=mode)
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.data[index]
        instance = {
            'image': data[0],
            'label': data[1] if self.mode!='test' else None,
            'file_id': data[2]
        } # numpy array
        return instance

    def collate_fn(self, samples):
        batch = {
            'image': torch.cat([self.transform(sample['image']).unsqueeze(0) for sample in samples]),
            'label': torch.tensor([sample['label'] for sample in samples]) if self.mode!='test' else None,
            'file_id': [sample['file_id'] for sample in samples]
        }   # torch tensor

        return batch


    def _parse_data(self, data_path, label_path, mode):

        if mode != 'test':
            labels = {}
            f = open(label_path, 'r')
            raw_labels = f.readlines()[1:]  # skip header

            for raw_label in raw_labels:
                file_name, label = raw_label.split(",")
                labels[file_name] = int(label.strip())
            f.close()

        data_list = []
        file_list = os.listdir(data_path)
        if self.mode == 'train':
            file_list = file_list[: len(file_list)*9//10]
        elif self.mode == 'val':
            file_list = file_list[len(file_list)*9//10: ]

        for ii, file_name in enumerate(file_list):
            img_path = os.path.join(data_path, file_name)
            img = np.array(Image.open(img_path).convert("RGB"))  # 0~255 
            if self.gray2rgb:
                img = img[..., np.newaxis]
                img = np.concatenate(3*(img, ), axis=-1) # shape=[28, 28, 3]
            #print(img.shape)
            label = labels[file_name] if mode != 'test' else None
            data_list.append([img, label, file_name])
            print("Parsing data {}/{}".format(ii+1, len(file_list))+" "*10, end='\r')
            #if ii == 100: break

        return data_list


class FaceDataset(Dataset):
    def __init__(self, data_path, label_path, mode='train'):

        assert mode in ('train', 'val', 'test'), f'Invalid mode={mode} given.'
        self.mode = mode    # valid options: 'train', 'val', 'test'
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        # list of [img, file_name]
        self.data = self._parse_data(self.data_path)
        # lables (dict): (k, v)=(file_name, attributes(list))
        self.labels = self._parse_labels(self.label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, file_name = self.data[index]
        instance = {
            'img':      img,
            'label':    np.array(self.labels[file_name]) if self.mode != 'test' else None,
            'file_name': file_name
        }   # numpy array

        return instance

    def collate_fn(self, samples):
        batch = {
            'img': torch.cat([self.transform(sample['img']).unsqueeze(0) for sample in samples]),
            'label': torch.tensor([sample['label'] for sample in samples]) if self.mode != 'test' else None,
            'file_name': [sample['file_name'] for sample in samples]
        }   # torch tensor

        return batch

    def _parse_labels(self, data_path):
        '''
        Parse data from `data_path`, which should be a csv file
        Args:
            data_path (str): path to which label is stored. 

        Return:
            data_list (dict): label dictionary (key=filename, value=attributes(np.array))
        '''
        if not data_path or not os.path.exists(data_path):
            print("Data path does not exist! path:", data_path)
            return []

        with open(data_path, 'r') as f:
            info_list = f.readlines()

        label_list = {}
        for info in info_list[1:]:  # skip header
            labels = info.split(",")
            file_name = labels[0][:-4]
            attributes = np.array([float(x) for x in labels[1:]])
            label_list[file_name] = attributes

        return label_list

    def _parse_data(self, data_path):
        '''
        Parse data from `data_path`. The folder contains only images.
        Args:
            data_path (str): path to which data is stored
        Return:
            data_list (list): list of parsed data in the format [image, file_name]
        '''
        if not os.path.exists(data_path):
            print("Data path does not exist! path:", data_path)
            return []

        file_list = os.listdir(data_path)
        # file_list.sort()

        data_list = []
        for i, file_name in enumerate(file_list):
            img_path = os.path.join(data_path, file_name)
            img = np.array(Image.open(img_path).convert("RGB"))  # PIL > 1.6
            data_list.append([img, file_name[:-4]])
            print("Parsing data {}/{}".format(i+1,
                                              len(file_list))+" "*50, end='\r')
            #if i == 100: break
        print("Finish parsing data." + " "*50)

        return data_list


if __name__ == '__main__':
    # dset = FaceDataset(
    #     data_path=os.path.join("hw3_data", "face", "train"), 
    #     label_path=os.path.join("hw3_data", "face", "train.csv"), 
    #     mode='train'
    # )
    domain = 'mnistm'
    dset = DigitDataset(
        data_path=os.path.join("hw3_data", "digits", domain, 'train'), 
        label_path=os.path.join("hw3_data", "digits", domain, "train.csv"), 
        mode='train'
    )
    print(dset[0])

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset=dset,
        batch_size=32,
        shuffle=True,
        collate_fn=dset.collate_fn
    )
    print(next(iter(train_loader)))

