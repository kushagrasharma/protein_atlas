import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from models import NeuralNetwork
from utils import ProteinAtlasTrainDataset

from datetime import datetime
now = datetime.now() 

train_timestr = now.strftime("%m-%d-%y-%H%M")

train_dir = 'data/train/'
train_csv_file = os.path.join(train_dir, 'train.csv')

def load_data():
    trainset = ProteinAtlasTrainDataset.ProteinAtlasTrainDataset(csv_file=train_csv_file, root_dir=train_dir, image_mean=13.42)

    train_loader = DataLoader(
        trainset,
        batch_size=16,
        num_workers=0,
        shuffle=True
    )

    return trainset, train_loader

def train_net(train_loader):

    lr = .0001

    PATH = 'models/protein_net_{}_lr_{:.6f}.pth'.format(train_timestr, lr)
    net = NeuralNetwork.NeuralNetwork()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    lambda1 = lambda epoch: max(0.35 ** epoch, .0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            image, labels = data['image'].unsqueeze(1), data['localizations']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f lr: %.6f' %
                      (epoch + 1, i + 1, running_loss / 20, optimizer.param_groups[0]["lr"]))
                running_loss = 0.0

            if i % 200 == 199:
                print("saved model")
                scheduler.step()
                torch.save(net.state_dict(), PATH)

    print('Finished Training')

    torch.save(net.state_dict(), PATH)

trainset, train_loader = load_data()

train_net(train_loader)