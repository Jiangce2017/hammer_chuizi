import os.path as osp
import os
from pathlib import Path
import numpy as np
import pickle
from numpy import linalg as LA
from models import FNO3d, r2loss, get_window
from general_FNO.MatDataset import BurgersDataset

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from applications.utilities3 import *
from timeit import default_timer
import csv
torch.manual_seed(0)
np.random.seed(0)

data_dir = os.path.join(Path.home(), 'data','burgers') 

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
    train_mse_center = 0
    num_sample = 0

    for data in train_loader:

        x_train, y_train = get_window(sample_region,world_res,ni,mi,li,w_radius,w_size,batch_size,x_train,y_train)
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        rho = x_train[:,:,:,:,[-1]]
        optimizer.zero_grad()
        pred_train = model(x_train).view(batch_size, w_size, w_size, w_size,1)
        pred_train *= rho
        y_train *= rho
        
        mse = F.mse_loss(pred_train, y_train, reduction='mean')
        l2 = myloss(pred_train.reshape(batch_size, -1), y_train.reshape(batch_size, -1))
        r2 = r2loss(pred_train.reshape(batch_size, -1), y_train.reshape(batch_size, -1))

        pred_train_center = pred_train[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
        y_train_center = y_train[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
        mse_center = F.mse_loss(pred_train_center, y_train_center, reduction='mean')

        l2.backward()
        optimizer.step()
        scheduler.step()

        train_mse += mse.item()
        train_l2 += l2.item()
        train_r2 += r2.item()
        train_mse_center += mse_center.item()
        num_sample += 1
        
    train_mse /= num_sample
    train_l2 /= num_sample
    train_r2 /= num_sample
    train_mse_center /= num_sample
    return train_mse,train_l2,train_r2,train_mse_center

def test(test_loader):
    model.eval()              
    test_mse = 0.0
    test_l2 = 0.0
    test_r2 = 0.0     
    test_mse_center = 0.0
    num_sample = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = get_window(sample_region,world_res,ni,mi,li,\
                                            w_radius,w_size,batch_size,x_test,y_test)
            x_test, y_test = x_test.to(device), y_test.to(device)
        
            rho = x_test[:,:,:,:,[-1]]
            pred_test = model(x_test).view(batch_size, w_size, w_size, w_size,1)     
            
            pred_test *= rho
            y_test *= rho
            
            test_mse += F.mse_loss(pred_test, y_test, reduction='mean').item()             
            
            test_l2 += myloss(pred_test.view(batch_size, -1), y_test.view(batch_size, -1)).item()
            test_r2 += r2loss(pred_test.view(batch_size, -1), y_test.view(batch_size, -1)).item()

            pred_test_center = pred_test[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
            y_test_center = y_test[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
            mse_center = F.mse_loss(pred_test_center, y_test_center, reduction='mean')

            test_mse_center += mse_center.item()
            num_sample += 1
    
    test_mse /= num_sample
    test_l2 /= num_sample
    test_r2 /= num_sample
    test_mse_center /= num_sample
    return test_mse,test_l2,test_r2,test_mse_center

modes = 8 # number of frequency modes
width = 20 # dimension of latent space
batch_size = 16
learning_rate = 0.001
epochs = 200

results_dir = osp.join('./','results')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = BurgersDataset(root=data_dir)
train_dataset, test_dataset = train_test_split(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

iterations = epochs * (len(train_loader) // batch_size)

model = FNO3d(modes=modes,width=width).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

cur_ep = 0

myloss = LpLoss(size_average=False)
train_logger = Logger(osp.join(results_dir, 'train.log'), ['epoch', 'train_mse', 'train_l2', 'train_r2', 'train_mse_center', \
                                                            'test_mse', 'test_l2', 'test_r2', 'test_mse_center'])

for ep in range(cur_ep, epochs):
    start_time = default_timer()
    train_mse,train_l2,train_r2,train_mse_center = train()
    test_mse,test_l2,test_r2,test_mse_center = test(test_loader)
    end_time = default_timer()
    epoch_time = end_time - start_time
    print('Epoch {}, time {:.4f}'.format(ep, epoch_time))
    print('train_mse: {:.4f}, train_l2: {:.4f}, train_r2: {:.4f}, train_mse_center: {:.4f}'.format(train_mse, train_l2, train_r2, train_mse_center))
    print('test_mse: {:.4f}, test_l2: {:.4f}, test_r2: {:.4f}, test_mse_center: {:.4f}'.format(test_mse, test_l2, test_r2, test_mse_center))
    train_logger.log({
        'epoch': ep,
        'train_mse': train_mse,
        'train_l2': train_l2,
        'train_r2': train_r2,
        'train_mse_center': train_mse_center,
        'test_mse': test_mse,
        'test_l2': test_l2,
        'test_r2': test_r2,
        'test_mse_center': test_mse_center,
    })

    if ep % 10 == 0:
        torch.save(model.state_dict(), osp.join(results_dir, 'model_ep{}.pth'.format(ep)))