import os

import torch
from torchvision import datasets, transforms

from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from skimage import io, transform

import warnings
warnings.filterwarnings("ignore")

class ProteinAtlasTrainDataset(Dataset):
    """Protein Atlas dataset."""

    def __init__(self, csv_file, root_dir, image_mean, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.localizations = pd.read_csv(csv_file)
        
        self.localizations.columns = ['id', 'labels']

        for i in range(28):
            self.localizations[i] = 0

        self.localizations.labels = self.localizations.labels.apply(lambda x: [int(y) for y in x.split(' ')])

        for i in range(len(self.localizations)):
            labels = self.localizations.iloc[i].labels
            for label in labels:
                self.localizations.loc[i, label] = 1
                
        self.root_dir = root_dir
        self.transform = transform
        self.image_mean = image_mean

    def __len__(self):
        return len(self.localizations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.localizations.loc[idx, 'id'] + '_green.png')
        image = io.imread(img_name)
        labels = self.localizations.iloc[idx,2:]
        labels = np.array(labels).astype(int)
        sample = {'image': torch.Tensor(image), 'localizations': torch.Tensor(labels)}
        sample['image'] = sample['image'] - self.image_mean

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
    