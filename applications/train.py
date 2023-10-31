### Train ML model
"""
@author: Jiangce Chen
This file is the Small Region Fourier Neural Operator for AM thermal simulation [paper](),
"""

import os.path as osp
import os
from pathlib import Path
import numpy as np
import pickle
from numpy import linalg as LA
from hammer import preprocess_data
from hammer import FNO3d,CNN

import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import csv
torch.manual_seed(0)
np.random.seed(0)

data_dir = os.path.join(Path.home(), 'data','hammer') 

################################################################
# 3d fourier layers
################################################################
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


def r2loss(pred, y):
    #print(pred.shape,y.shape)
    SS_res = torch.sum((pred-y)**2,dim=1)
    y_mean = torch.mean(y,dim=1,keepdims=True)
    SS_tot = torch.sum((y-y_mean)**2,dim=1)
    r2 = 1 - SS_res/SS_tot
    return torch.mean(r2,dim=0)
    
def mseloss(pred,y):
    mse = torch.mean((pred-y)**2,dim=1)
    return torch.mean(mse)

################################################################
# configs
################################################################

w_size = 11

resolution = 20
num_cut = 10
bjorn = True
#preprocess_data(w_size,model_name,resolution,bjorn)

modes = 8 # 8
width = 20 #20

batch_size = 16
learning_rate = 0.001
epochs = 50

results_dir = osp.join('./','results')

runtime = np.zeros(2, )
t1 = default_timer()



sub = 1
S = w_size // sub
T_in = w_size
T = w_size # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;

################################################################
# load data
################################################################
total_model_names = ['townhouse_2','townhouse_3','townhouse_5','townhouse_6','townhouse_7','hollow_1','hollow_2','hollow_3','hollow_4','hollow_5',]
for valid_model_name in total_model_names:
    print(valid_model_name)
    s = set([valid_model_name])
    model_names = [x for x in total_model_names if x not in s]
    print(model_names)
    a = []
    u = []
    
    for model_name in model_names:
        data_path = osp.join(data_dir,"ml_data",model_name,model_name+'_cut'+str(num_cut)+'.pk')
        data = pickle.load(open( data_path, "rb" ))
        a.append(data["a"])
        u.append(data["u"])

    a = np.concatenate(a,axis=0)
    u = np.concatenate(u,axis=0)
    num_data = a.shape[0]
    idx = np.random.permutation(range(num_data))
    a, u = a[idx,:,:,:,:], u[idx,:,:,:,:]

    a = torch.from_numpy(a)
    u = torch.from_numpy(u)
    test_per = 0.1
    ntest = np.round(test_per*num_data).astype(np.int32)
    test_a, test_u = a[:ntest,:,:,:,:], u[:ntest,:,:,:,:]
    train_a, train_u = a[ntest:,:,:,:,:], u[ntest:,:,:,:,:]

    ntrain = num_data-ntest
    iterations = epochs*(ntrain//batch_size)

    print(train_a.shape)
    print(test_a.shape)
    print(train_u.shape)
    print(test_u.shape)
    assert (S == train_u.shape[-2])
    assert (1 == train_u.shape[-1])

    #### normalize the data
    a_normalizer = UnitGaussianNormalizer(train_a[:,:,:,:,:-1])
    train_a[:,:,:,:,:-1] = a_normalizer.encode(train_a[:,:,:,:,:-1])
    test_a[:,:,:,:,:-1] = a_normalizer.encode(test_a[:,:,:,:,:-1])

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)
    test_u = y_normalizer.encode(test_u)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, drop_last = True)

    t2 = default_timer()


    #### valid model 
    valid_data_path = osp.join(data_dir,"ml_data",valid_model_name,valid_model_name+'_cut'+str(num_cut)+'.pk')
    valid_data = pickle.load(open( valid_data_path, "rb" ))
    valid_a = valid_data["a"]
    valid_u = valid_data["u"]
    valid_a = torch.from_numpy(valid_a)
    valid_u = torch.from_numpy(valid_u)
    valid_a[:,:,:,:,:-1] = a_normalizer.encode(valid_a[:,:,:,:,:-1])
    valid_u = y_normalizer.encode(valid_u)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_a, valid_u), batch_size=batch_size, shuffle=True, drop_last = True)

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda:0')
    cpu_device = torch.device('cpu')

    ################################################################
    # training and evaluation
    ################################################################
    #model = FNO3d(modes, modes, modes, width)
    model = CNN(modes, modes, modes, 128)
    model = model.to(device)
    print(count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    scope = 'local'
    exp_name = valid_model_name+'_CNN'+'_'+scope
    path = exp_name+'_ep' + str(epochs) + '_m' + str(modes) \
        + '_w' + str(width) + '_window_sz' + str(w_size) + '_cut'+str(num_cut) + valid_model_name
    path_model = 'model/'+path

    cur_ep = 0
    # if os.path.isfile(path_model):
    #     checkpoint = torch.load(path_model)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     cur_ep = checkpoint['epoch']

    myloss = LpLoss(size_average=False)
    y_normalizer = y_normalizer.to(device)
    train_logger = Logger(
        osp.join(results_dir, exp_name+'_train.log'),
        ['ep', 'train_mse','train_l2','train_r2','test_mse','test_l2','test_r2','valid_mse','valid_l2','valid_r2']
    )
    for ep in range(cur_ep+1,epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        train_r2 = 0.0
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            rho = x_train[:,:,:,:,[-1]]
            optimizer.zero_grad()
            pred_train = model(x_train).view(batch_size, S, S, T,1)
            pred_train *= rho
            y_train *= rho

            y_train = y_normalizer.decode(y_train)
            pred_train = y_normalizer.decode(pred_train)

            l2 = myloss(pred_train.view(batch_size, -1), y_train.view(batch_size, -1))
            l2.backward()
            optimizer.step()
            scheduler.step()

            train_mse += F.mse_loss(pred_train, y_train, reduction='mean').item()
            train_l2 += l2.item()
            train_r2 += r2loss(pred_train.view(batch_size, -1), y_train.view(batch_size, -1)).item()

        model.eval()
        test_mse = 0.0
        test_l2 = 0.0
        test_r2 = 0.0
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                rho = x_test[:,:,:,:,[-1]]
                pred_test = model(x_test).view(batch_size, S, S, T,1)
                pred_test *= rho
                y_test *= rho
                
                y_test = y_normalizer.decode(y_test)
                pred_test = y_normalizer.decode(pred_test)
                
                test_mse += F.mse_loss(pred_test, y_test, reduction='mean').item()
                test_l2 += myloss(pred_test.view(batch_size, -1), y_test.view(batch_size, -1)).item()
                test_r2 += r2loss(pred_test.view(batch_size, -1), y_test.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= len(train_loader)
        train_r2 /= len(train_loader)
        
        test_mse /= len(test_loader)
        test_l2 /= len(test_loader)
        test_r2 /= len(test_loader)

        t2 = default_timer()

        model.eval()
        valid_mse = 0.0
        valid_l2 = 0.0
        valid_r2 = 0.0
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                rho = x_valid[:,:,:,:,[-1]]
                pred_valid = model(x_valid).view(batch_size, S, S, T,1)     
        
                pred_valid *= rho
                y_valid *= rho
                
                y_valid = y_normalizer.decode(y_valid)
                pred_valid = y_normalizer.decode(pred_valid)
                
                valid_mse += F.mse_loss(pred_valid, y_valid, reduction='mean').item()
                valid_l2 += myloss(pred_valid.view(batch_size, -1), y_valid.view(batch_size, -1)).item()
                valid_r2 += r2loss(pred_valid.view(batch_size, -1), y_valid.view(batch_size, -1)).item()


        valid_mse /= len(valid_loader)
        valid_l2 /= len(valid_loader)
        valid_r2 /= len(valid_loader)

        print(f'ep: {ep}, train mse: {train_mse:.6f},train L2: {train_l2:.6f},train r2: {train_r2:.6f}, \
              test mse: {test_mse:.6f}, test L2: {test_l2:.6f},test r2: {test_r2:.6f},\
                valid mase: {valid_mse}, valid l2:{valid_l2}, valid r2:{valid_r2}')
        train_logger.log({
            'ep': ep,           
            'train_mse': train_mse,
            'train_l2': train_l2,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_l2': test_l2,
            'test_r2': test_r2,
            'valid_mse':valid_mse,
            'valid_l2':valid_l2,
            'valid_r2':valid_r2
        })
        torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'test_l2':test_l2,
                    }, path_model)
                    
    model.eval()
    valid_mse = 0.0
    valid_l2 = 0.0
    valid_r2 = 0.0
    input_a = []
    true_u = []
    pred_u = []
    a_normalizer.to(device)
    with torch.no_grad():
        for x_valid, y_valid in valid_loader:
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            rho = x_valid[:,:,:,:,[-1]]
            pred_valid = model(x_valid).view(batch_size, S, S, T,1)     
    
            pred_valid *= rho
            y_valid *= rho
            
            y_valid = y_normalizer.decode(y_valid)
            pred_valid = y_normalizer.decode(pred_valid)
            
            valid_mse += F.mse_loss(pred_valid, y_valid, reduction='mean').item()
            valid_l2 += myloss(pred_valid.view(batch_size, -1), y_valid.view(batch_size, -1)).item()
            valid_r2 += r2loss(pred_valid.view(batch_size, -1), y_valid.view(batch_size, -1)).item()

            x_valid[:,:,:,:,:-1] = a_normalizer.decode(x_valid[:,:,:,:,:-1])
            input_a.append(x_valid.to(cpu_device))
            true_u.append(y_valid.to(cpu_device))
            pred_u.append(pred_valid.to(cpu_device))

    valid_mse /= len(valid_loader)
    valid_l2 /= len(valid_loader)
    valid_r2 /= len(valid_loader)
    print(f'valid mse: {valid_mse:.6f},valid L2: {valid_l2:.6f},valid r2: {valid_r2:.6f}')

    input_a = np.concatenate(input_a, axis=0)
    true_u = np.concatenate(true_u, axis=0)
    pred_u = np.concatenate(pred_u, axis=0)
    data = {
        "a":input_a,
        "true_u":true_u,
        "pred_u": pred_u,
        "window_info": valid_data["window_info"]
    }
    pickle.dump(data, open( osp.join(results_dir, 'results'+exp_name+'_cut'+str(num_cut)+".pk"), "wb" ))




    