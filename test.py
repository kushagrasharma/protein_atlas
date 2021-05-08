import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

import numpy as np
import pandas as pd

from models import NeuralNetwork
from utils import ProteinAtlasTestDataset

PATH = 'models/protein_net_05-08-21-1421_lr_0.000100.pth'

test_dir = 'data/test/'
test_csv_file = os.path.join(test_dir, 'sample_submission.csv')

testset = ProteinAtlasTestDataset.ProteinAtlasTestDataset(test_csv_file, test_dir, image_mean=13.42)

test_loader = DataLoader(
    testset,
    batch_size=64,
    num_workers=0,
    shuffle=False
)

net = NeuralNetwork.NeuralNetwork()

net.load_state_dict(torch.load(PATH))
net.eval()

prediction_id = []
prediction_label = []

for i, data in enumerate(test_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    image, image_id = data['image'].unsqueeze(1), data['id']

    # forward + backward + optimize
    outputs = net(image)

    array = outputs.detach().numpy()
    array = np.around(array).astype(int)

    for i in range(array.shape[0]):
        labels = np.argwhere(array[i])
        if not len(labels):
            labels = str(np.argmax(array[i]))
        else:
            labels = reversed([str(item) for sublist in labels for item in sublist])
            labels = ' '.join(labels)
        prediction_id.append(image_id[i])
        prediction_label.append(labels)

submission = pd.DataFrame({'Id' : prediction_id, 'Predicted' : prediction_label})
submission.to_csv("submission.csv", index=False)