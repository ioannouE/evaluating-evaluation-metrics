import h5py
import torch
import numpy as np
import os
from PIL import Image
import pandas as pd
from torchvision import transforms



class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, path_to_h5_file, transforms=None):
        """
        PyTorch dataset for images that are stored in a h5 file.

        Args:
            path_to_h5_file (str): Path to h5 file
            transforms (albumentations.Compose): Augmentations
        """
        with h5py.File(path_to_h5_file, 'r') as f:
            self.length = f['images'].shape[0]
        #self.h5_file = h5py.File(path_to_h5_file, 'r')
        self.path_to_h5_file = path_to_h5_file
        self.transforms = transforms

    def __len__(self):
        #return self.h5_file['images'].shape[0]
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.path_to_h5_file, 'r')
        img = np.array(self.h5_file['images'][idx], dtype=np.uint8)
        img = Image.fromarray(img)
        artist_label = self.h5_file['artist_labels'][idx]
        style_label = self.h5_file['style_labels'][idx]

        if self.transforms is not None:
            #img = self.transforms(image=img)['image']
            img = self.transforms(img)

        sample = {'image': img, 'artist_label': artist_label, 'style_label': style_label} 
        return sample


class ImageFolder(torch.utils.data.Dataset):

    def __init__(self, path, transforms=None):
        """
        PyTorch dataset for images that are stored in a directory.

        Args:
            path (str): Path to image directory 
            transforms (albumentations.Compose): Augmentations
        """
        self.img_names = os.listdir(path)
        self.path = path
        self.transforms = transforms

    def __len__(self):
        return len(self.img_names) 

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.img_names[idx]))
        img = np.array(img, dtype=np.float32)

        if self.transforms is not None:
            img = self.transforms(image=img)['image'] 
        return img


# custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, artist_label_dict, style_label_dict, transforms=None):
        self.data = pd.read_csv(csv_file, encoding='utf8')
        self.img_dir = img_dir
        self.artist_label_dict = artist_label_dict
        self.style_label_dict = style_label_dict
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if (self.data.iloc[idx,4] == 'downloaded'):
            img_path = self.data.iloc[idx, 3]
            
            # img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            # image = np.array(image, dtype=np.float32)

            artist_label = self.data.iloc[idx, 0].strip()
            artist_label = self.artist_label_dict[artist_label]
            artist_label = torch.tensor(artist_label, dtype=torch.float32)
            artist_label = artist_label.type(torch.LongTensor)

            style_label = self.data.iloc[idx, 1].strip()
            style_label = self.style_label_dict[style_label]
            style_label = torch.tensor(style_label, dtype=torch.float32)    
            style_label = style_label.type(torch.LongTensor)
            
            if self.transforms:
                image = self.transforms(image)
            
            sample = {'image': image, 'artist_label': artist_label, 'style_label': style_label} 
            return sample