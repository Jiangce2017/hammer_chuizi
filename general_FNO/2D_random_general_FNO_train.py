import os.path as osp
import os
from pathlib import Path
import numpy as np
import pickle
from numpy import linalg as LA
from models import r2loss
from FNO_2d import FNO2d
from MatDataset import BurgersDataset

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from utilities3 import *
from timeit import default_timer
import csv
torch.manual_seed(0)
np.random.seed(0)

data_dir = os.path.join('general_FNO', 'data','burgers') 

def nrmseloss(pred,y):
    mse = torch.mean((pred-y)**2,dim=1)
    return torch.mean(mse)
    
def mseloss(pred,y):
    mse = torch.mean((pred-y)**2,dim=1)
    return torch.mean(mse)

def train_test_split(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
    
class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def train():
    model.train()
    train_mse = 0.0
    train_l2 = 0.0
    train_r2 = 0.0

    for data in train_loader:

        x, y = data[0], data[1]
        x = x.to(device)
        y = y.to(device)

        pred = model(x).view(batch_size, window_size, window_size , 1)

        train_mse += F.mse_loss(pred, y, reduction='mean').item()
        train_l2 += myloss(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
        train_r2 += r2loss(pred.view(batch_size, -1), y.view(batch_size, -1)).item()

        optimizer.zero_grad()
        loss = F.mse_loss(pred, y, reduction='mean')
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_mse /= len(train_loader)
    train_l2 /= len(train_loader)
    train_r2 /= len(train_loader)

    return train_mse, train_l2, train_r2

def test(test_loader):
    model.eval()              
    test_mse = 0.0
    test_l2 = 0.0
    test_r2 = 0.0     
    with torch.no_grad():
        for data in test_loader:
            x, y = data[0], data[1]
            x = x.to(device)
            y = y.to(device)
            pred = model(x).view(batch_size, window_size, window_size , 1)
            test_mse += F.mse_loss(pred, y, reduction='mean').item()
            test_l2 += myloss(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
            test_r2 += r2loss(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
    
    test_mse /= len(test_loader)
    test_l2 /= len(test_loader)
    test_r2 /= len(test_loader)

    return test_mse, test_l2, test_r2

modes = 8 # number of frequency modes
width = 20 # dimension of latent space
batch_size = 16
learning_rate = 0.001
epochs = 200
window_size = 8

results_dir = 'general_FNO/results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = BurgersDataset(root=data_dir)
train_dataset, test_dataset = train_test_split(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

iterations = epochs * (len(train_loader) // batch_size)

model = FNO2d(modes1=modes, modes2=modes, width=width).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

cur_ep = 0

myloss = LpLoss(size_average=False)
train_logger = Logger(osp.join(results_dir, 'train.log'), ['epoch', 'train_mse', 'train_l2', 'train_r2', \
                                                            'test_mse', 'test_l2', 'test_r2'])

for ep in range(cur_ep, epochs):
    start_time = default_timer()
    train_mse, train_l2, train_r2 = train()
    test_mse, test_l2, test_r2 = test(test_loader)
    end_time = default_timer()
    epoch_time = end_time - start_time
    print('Epoch {}, time {:.4f}'.format(ep, epoch_time))
    print('train_mse: {:.4f}, train_l2: {:.4f}, train_r2: {:.4f}'.format(train_mse, train_l2, train_r2))
    print('test_mse: {:.4f}, test_l2: {:.4f}, test_r2: {:.4f}'.format(test_mse, test_l2, test_r2))
    train_logger.log({
        'epoch': ep,
        'train_mse': train_mse,
        'train_l2': train_l2,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_l2': test_l2,
        'test_r2': test_r2,
    })

    if ep % 10 == 0:
        torch.save(model.state_dict(), osp.join(results_dir, 'model_ep{}.pth'.format(ep)))