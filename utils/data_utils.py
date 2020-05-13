import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from conf import config
from utils.data_augmentation import FixedRotation


class MyDataset(Dataset):
    def __init__(self, df, transform, mode='train'):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(self.df['FileID'].iloc[index]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(np.array(self.df['SpeciesID'].iloc[index]))
        else:
            img = Image.open(self.df[index]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(np.array(0))

    def __len__(self):
        return len(self.df)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.RandomRotation(15),
    transforms.RandomChoice([transforms.Resize([256, 256]), transforms.CenterCrop([256, 256])]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize([config.img_size, config.img_size]),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize([config.img_size, config.img_size]),
    transforms.ToTensor(),
    normalize,
])
