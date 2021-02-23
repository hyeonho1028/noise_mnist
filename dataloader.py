import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 


class PathDataset(Dataset): 
    def __init__(self, image_paths, labels=None, default_transforms=None, transforms=None, is_test=False): 
        self.image_paths = image_paths
        self.labels = labels 
        self.default_transforms = default_transforms
        self.transforms = transforms
        self.is_test = is_test

        self.imgs = []

        # for img_path in self.image_paths:
        #     img = cv2.imread(img_path)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     img = Image.fromarray(img)

        #     if self.default_transforms is not None:
        #         img = self.default_transforms(img)
    
        #     self.imgs.append(img)

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.default_transforms is not None:
            img = self.default_transforms(img)

        # img = self.imgs[index]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.is_test:
            return torch.tensor(img, dtype=torch.float32)
        else:
            return torch.tensor(img, dtype=torch.float32),\
                 torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self): 
        return len(self.image_paths)

