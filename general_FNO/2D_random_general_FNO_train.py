import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
# from models import r2loss
from FNO_2d import FNO2d, DomainPartitioning2d
from MatDataset import BurgersDataset, BurgersDatasetWhole

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from utilities3 import *
from timeit import default_timer
import csv
import wandb
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

def plot_prediction(window_size, y, y_pred, epoch, batch_idx, folder):
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].contourf(xx, yy, y.cpu().detach().numpy().reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[0].set_title('Ground truth')
    axs[0].axis('off')
    axs[1].contourf(xx, yy, y_pred.cpu().detach().numpy().reshape(window_size, window_size), levels=100, cmap='plasma')
    axs[1].set_title('Prediction')
    axs[1].axis('off')
    axs[2].contourf(xx, yy, np.abs(y.cpu().detach().numpy().reshape(window_size, window_size) - y_pred.cpu().detach().numpy().reshape(window_size, window_size)), levels=100, cmap='plasma')
    axs[2].set_title('Absolute difference')
    axs[2].axis('off')
    # plt.savefig(osp.join(folder, 'ep{}_sub_batch{}.png'.format(epoch, batch_idx)))
    # plt.close()
    wandb.log({"prediction": wandb.Image(axs[1])})


def train_sub_domain(train_loader):
    model.train()
    train_mse = 0.0
    train_l2 = 0.0
    train_r2 = 0.0
    reconstructed_r2 = 0.0

    for data in train_loader:

        x, y = data[0], data[1]
        sub_x_list = model.get_partition_domain(x, mode='train')
        sub_y_list = model.get_partition_domain(y, mode='test')
        pred_list = []
        for i in range(len(sub_x_list)):
            sub_x = sub_x_list[i].to(device)
            sub_y = sub_y_list[i].to(device)

            pred = model(sub_x)
            pred_list.append(pred)

            train_mse += F.mse_loss(pred, sub_y, reduction='mean').item()
            # train_l2 += myloss(pred.view(batch_size, -1), sub_y.view(batch_size, -1)).item()
            train_l2 += myloss(pred, sub_y).item()
            train_r2 += r2_score(sub_y.cpu().detach().numpy().reshape(batch_size, -1), pred.cpu().detach().numpy().reshape(batch_size, -1))

            optimizer.zero_grad()
            loss = F.mse_loss(pred, sub_y, reduction='mean')
            loss.backward()
            optimizer.step()
            scheduler.step()

        pred_y = model.reconstruct_from_partitions(y, pred_list)
        reconstructed_r2 += r2_score(y.cpu().detach().numpy().reshape(batch_size, -1), pred_y.cpu().detach().numpy().reshape(batch_size, -1))

    train_mse /= (len(train_loader) * len(sub_x_list))
    train_l2 /= (len(train_loader) * len(sub_x_list))
    train_r2 /= (len(train_loader) * len(sub_x_list))
    reconstructed_r2 /= len(train_loader)

    return train_mse, train_l2, train_r2


def train(train_loader):
    model.train()
    train_mse = 0.0
    train_l2 = 0.0
    train_r2 = 0.0

    for data in train_loader:

        x, y = data[0], data[1]
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        train_mse += F.mse_loss(pred, y, reduction='mean').item()
        train_l2 += myloss(pred.view(batch_size, -1), y.view(batch_size, -1)).item()
        train_r2 += r2_score(y.cpu().detach().numpy().reshape(batch_size, -1), pred.cpu().detach().numpy().reshape(batch_size, -1))

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
    reconstructed_r2 = 0.0
    with torch.no_grad():
        for data in test_loader:
            x, y = data[0], data[1]
            sub_x_list = model.get_partition_domain(x)
            sub_y_list = model.get_partition_domain(y)
            pred_list = []
            for sub_x, sub_y in sub_x_list, sub_y_list:
                sub_x = sub_x.to(device)
                sub_y = sub_y.to(device)
                pred = model(sub_x)
                pred_list.append(pred)
                test_mse += F.mse_loss(pred, sub_y, reduction='mean').item()
                test_l2 += myloss(pred.view(batch_size, -1), sub_y.view(batch_size, -1)).item()
                test_r2 += r2_score(sub_y.cpu().detach().numpy().reshape(batch_size, -1), pred.cpu().detach().numpy().reshape(batch_size, -1))
        pred_y = model.reconstruct_from_partitions(pred_list)
        reconstructed_r2 += r2_score(y.cpu().detach().numpy().reshape(batch_size, -1), pred_y.reshape(batch_size, -1))

    test_mse /= (len(test_loader) * len(sub_x_list))
    test_l2 /= (len(test_loader) * len(sub_x_list))
    test_r2 /= (len(test_loader) * len(sub_x_list))
    reconstructed_r2 /= len(test_loader)

    return test_mse, test_l2, test_r2, reconstructed_r2


def test_sub_domain(test_loader):
    model.eval()              
    test_mse = 0.0
    test_l2 = 0.0
    test_r2 = 0.0    
    reconstructed_r2 = 0.0 
    with torch.no_grad():
        for data in test_loader:
            x, y = data[0], data[1]
            sub_x_list = model.get_partition_domain(x, mode='train')
            sub_y_list = model.get_partition_domain(y, mode='test')
            pred_list = []
            for i in range(len(sub_x_list)):
                sub_x = sub_x_list[i].to(device)
                sub_y = sub_y_list[i].to(device)
                pred = model(sub_x)
                pred_list.append(pred)
                test_mse += F.mse_loss(pred, sub_y, reduction='mean').item()
                # test_l2 += myloss(pred.view(batch_size, -1), sub_y.view(batch_size, -1)).item()
                test_r2 += r2_score(sub_y.cpu().detach().numpy().reshape(batch_size, -1), pred.cpu().detach().numpy().reshape(batch_size, -1))
                test_l2 += myloss(pred, sub_y).item()
                # test_r2 += r2_score(sub_y.cpu().detach().numpy(), pred.cpu().detach().numpy())

            pred_y = model.reconstruct_from_partitions(y, pred_list)
            # test_r2 += r2_score(y.cpu().detach().numpy().reshape(batch_size, -1), pred_y.cpu().detach().numpy().reshape(batch_size, -1))
            reconstructed_r2 += r2_score(y.cpu().detach().numpy().reshape(batch_size, -1), pred_y.cpu().detach().numpy().reshape(batch_size, -1))

    test_mse /= (len(test_loader) * len(sub_x_list))
    test_l2 /= (len(test_loader) * len(sub_x_list))
    test_r2 /= (len(test_loader) * len(sub_x_list))
    reconstructed_r2 /= len(test_loader)

    return test_mse, test_l2, test_r2, reconstructed_r2

modes = 8 # number of frequency modes
width = 20 # dimension of latent space
batch_size = 16
learning_rate = 0.001
epochs = 200
window_size = 12

results_dir = 'general_FNO/results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="Domain_partition_2D", group="2d_burgers")

# dataset = BurgersDataset(root=data_dir)
dataset = BurgersDatasetWhole(root=data_dir)[:800]
# pick 0.5 of the dataset as data
# dataset = dataset[:int(len(dataset) * 0.5)]
train_dataset, test_dataset = train_test_split(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

iterations = epochs * (len(train_loader) // batch_size)

model = DomainPartitioning2d(modes, modes, width, window_size).to(device)
# model = FNO2d(modes1=modes, modes2=modes, width=width).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

cur_ep = 0

myloss = LpLoss(size_average=False)
# train_logger = Logger(osp.join(results_dir, 'train_sub_batch.log'), ['epoch', 'train_mse', 'train_r2', \
#                                                             'test_mse', 'test_r2', 'reconstructed_r2'])

for ep in range(cur_ep, epochs):
    start_time = default_timer()
    train_mse, train_l2, train_r2 = train_sub_domain(train_loader)
    # test_mse, test_l2, test_r2 = test(test_loader)
    # test_mse, test_l2, test_r2, reconstructed_r2 = test_sub_domain(test_loader)
    end_time = default_timer()
    epoch_time = end_time - start_time
    print('Epoch {}, time {:.4f}'.format(ep, epoch_time))
    print('train_mse: {:.4f}, train_r2: {:.4f}'.format(train_mse, train_r2))
    
    # train_logger.log({
    #     'epoch': ep,
    #     'train_mse': train_mse,
    #     'train_r2': train_r2,
    #     'test_mse': test_mse,
    #     'test_r2': test_r2,
    #     'reconstructed_r2': reconstructed_r2
    # })
    wandb.log({
        'train_mse': train_mse,
        'train_r2': train_r2,
    })

    if ep % 10 == 0:
        torch.save(model.state_dict(), osp.join(results_dir, 'model_ep{}.pth'.format(ep)))
        test_mse, test_l2, test_r2, reconstructed_r2 = test_sub_domain(test_loader)
        print('test_mse: {:.4f}, test_r2: {:.4f}'.format(test_mse, test_r2))
        wandb.log({
            'test_mse': test_mse,
            'test_r2': test_r2,
            'reconstructed_r2': reconstructed_r2
        })
        # plot prediction
        x, y = next(iter(test_loader))
        sub_x_list = model.get_partition_domain(x, mode='train')
        sub_y_list = model.get_partition_domain(y, mode='test')
        pred_list = []
        for i in range(len(sub_x_list)):
            sub_x = sub_x_list[i].to(device)
            sub_y = sub_y_list[i].to(device)
            pred = model(sub_x)
            pred_list.append(pred)
        pred_y = model.reconstruct_from_partitions(y, pred_list)
        # x = x.to(device)
        # y = y.to(device)
        # pred = model(x)
        plot_prediction(y[0].shape[0], y[0], pred_y[0], ep, 0, results_dir)
        plot_prediction(window_size, sub_y[0], pred[0], ep, 1, results_dir)
