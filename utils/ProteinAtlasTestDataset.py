import os

import torch
from torchvision import datasets, transforms

from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from skimage import io, transform

import warnings
warnings.filterwarnings("ignore")

class ProteinAtlasTestDataset(Dataset):
    """Protein Atlas dataset."""

    def __init__(self, csv_file, root_dir, image_mean, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.test_csv = pd.read_csv(csv_file)
        self.test_csv.columns = ['id', 'labels']

        self.test_csv['filenames'] = self.test_csv.id.apply(lambda x: os.path.join(self.root_dir, x + '_green.png'))

        self.transform = transform

        self.image_mean = image_mean

    def __len__(self):
        return len(self.test_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.test_csv.filenames.loc[idx]

        image = io.imread(img_name)
        sample = {'image': torch.Tensor(image), 'id': self.test_csv.id.loc[idx]}
        sample['image'] = sample['image'] - self.image_mean

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample