import os
import torch
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, data_path=os.path.join('hw2_data', 'p2_data'), mode='train'):
        self.data_path = data_path
        
        assert mode in ('train', 'val', 'test'), f'Invalid mode={mode} given.'
        self.mode = mode    # valid options: 'train', 'val', 'test'
        
        self.img_transforms = transforms.Compose([    
                #transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])
        
        
        # list of [img, label, file_id]
        self.data = self.parse_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'img':    sample[0],
            'label':  sample[1],
            'id':     sample[2]
        }   # numpy array

        return instance

    def collate_fn(self, samples):  
        batch = {
            'img': torch.cat([self.img_transforms(sample['img']).unsqueeze(0) for sample in samples]),
            'label': torch.tensor([sample['label'] for sample in samples]),
            'id': [sample['id'] for sample in samples]
        }   # torch tensor

        # unsqueeze(0): since we have [W, H, 3] originally, 
        # and using torch.cat will merge tensors in the first dimension,
        # we have to give the tensor an additional dimension: [1, W, H, 3]
        # This will give the correct batched form: [bs, W, H, 3]
        return batch

    def parse_data(self, data_path):

        if not os.path.exists(data_path):
            print("Data path does not exist! path:", data_path)
            return []

        file_list = os.listdir(data_path)
        #file_list.sort()
        file_list = set([file_name.split('_')[0] for file_name in file_list])
        
        
        data_list = []
        for i, file_id in enumerate(file_list):
            img_path = os.path.join(data_path, f"{file_id}_sat.jpg")
            label_path  = os.path.join(data_path, f"{file_id}_mask.png")
            img = Image.open(img_path).convert("RGB")

            if self.mode == 'train' or self.mode == 'val':
                mask_img = np.array(Image.open(label_path).convert("RGB"))
                label = self.convert_label(mask_img)
            else: # mode = 'test'
                label = np.zeros(img.size[0:2])

            data_list.append([img, label, file_id])
            print("Parsing data {}/{}".format(i+1, len(file_list))+" "*50, end='\r')
            
            #if i == 100: break
        print("Finish data parsing.")

        return data_list

    def convert_label(self, mask):
        mask = np.array(mask)
        label = np.empty(mask.shape[0: 2], dtype=np.int)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2] # binart to decimal
        label[mask == 3] = 0  # (Cyan: 011) Urban land
        label[mask == 6] = 1  # (Yellow: 110) Agriculture land
        label[mask == 5] = 2  # (Purple: 101) Rangeland
        label[mask == 2] = 3  # (Green: 010) Forest land
        label[mask == 1] = 4  # (Blue: 001) Water
        label[mask == 7] = 5  # (White: 111) Barren land
        label[mask == 0] = 6  # (Black: 000) Unknown
        label[mask == 4] = 6  # (Red: 100) Unknown

        return label


class ClassificationDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        self.data_path = data_path
        self.mode = mode
        self.img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        # list of [img, label, id]
        self.data = self.get_data(data_path)        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'img':    sample[0],
            'label':  sample[1],
            'id':     sample[2]
        }

        return instance

    def collate_fn(self, samples):
        batch = {
            'img': torch.cat([self.img_transforms(sample['img']).unsqueeze(0) for sample in samples]),
            'label': torch.tensor([sample['label'] for sample in samples]),
            'id': [sample['id'] for sample in samples]
        }
        # for key in ['img', 'label']:
        #    batch[key] = torch.cat([sample[key] for sample in samples])

        return batch

    def get_data(self, data_path):
        if not os.path.exists(data_path):
            print("Data path does not exist! path:", data_path)
            return []

        data_list = []
        file_list = os.listdir(data_path)
        for i, img_name in enumerate(file_list):
            img_path = os.path.join(data_path, img_name)
            img = Image.open(img_path).convert("RGB")
            
            if self.mode == 'train' or self.mode == 'val':
                label = int(img_name.split('_')[0])
            else:
                label = 0

            data_list.append([img, label, img_name])
            print("Parsing data {}/{}".format(i+1, len(file_list)), end='\r')
        print("Finishing parsing data.")   

        return data_list


    
if __name__ == '__main__':
    #train_dset = SegmentationDataset(data_path=os.path.join('hw2_data', 'p2_data', 'train'), mode='train')
    val_dset = SegmentationDataset(data_path=os.path.join('hw2_data', 'p2_data', 'validation'), mode='val')

    def test_dset(img, fname):
        (fname, img)
    
    import matplotlib.pyplot as plt
    #plt.imsave("test_train_dataset_img.png", train_dset[0]['img'])
    #plt.imsave("test_train_dataset_maks.png", train_dset[0]['label'])
    #plt.imsave("test_val_dataset_img.png", val_dset[0]['img'])
    #plt.imsave("test_val_dataset_mask.png", val_dset[0]['label'])
    print(val_dset[0]['img'].shape, val_dset[0]['label'].shape)