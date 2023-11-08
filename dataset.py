import os

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import cv2
import numpy as np


class BUSI(Dataset):
    def __init__(self, root, image_transform, mask_transform, target_transform=None):
        super().__init__()

        self.root = root

        self.images = []
        self.masks = []
        self.labels = []

        self.make_dataset()

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.target_transform = target_transform


    def make_dataset(self):
        for target in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if not path.endswith('mask.png'):
                        mask_path = path.replace('.png', '_mask.png')
                        if os.path.isfile(mask_path):
                            self.images.append(path)
                            self.masks.append(mask_path)
                            self.labels.append(target)


    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(self.labels[index])

        return image, mask, target


    def __len__(self):
        return len(self.images)
        

class CT2US(Dataset):
    def __init__(self, root, transforms):
        super().__init__()

        self.root = root

        self.images = []
        self.masks = []

        self.make_dataset()

        self.transforms = transforms

    
    def make_dataset(self):
        images_path = os.path.join(self.root, 'slice', 'slice')
        masks_path = os.path.join(self.root, 'mask', 'mask')

        for image_name in sorted(os.listdir(images_path)):
            image_path = os.path.join(images_path, image_name)
            mask_path = os.path.join(masks_path, image_name)

            if os.path.isfile(mask_path):
                self.images.append(image_path)
                self.masks.append(mask_path)

    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE).astype('float32') / 255
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE).astype('float32') / 255

        # unsqueeze to add a channel dimension
        image = np.expand_dims(image, axis=2)
        mask = np.expand_dims(mask, axis=2)

        image, mask = self.transforms(image, mask)

        return image, mask


    def __len__(self):
        return len(self.images)
