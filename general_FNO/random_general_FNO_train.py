### Train ML model
"""
@author: Jiangce Chen
"""

import os.path as osp
import os
from pathlib import Path
import numpy as np
import pickle
from numpy import linalg as LA
from models import FNO3d, r2loss, get_window

import torch.nn.functional as F
from applications.utilities3 import *
from timeit import default_timer
import csv
torch.manual_seed(0)
np.random.seed(0)

data_dir = os.path.join(Path.home(), 'data','hammer') 

    
def nrmseloss(pred,y):
    mse = torch.mean((pred-y)**2,dim=1)
    return torch.mean(mse)
    
def mseloss(pred,y):
    mse = torch.mean((pred-y)**2,dim=1)
    return torch.mean(mse)
    
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

    for x_train, y_train in train_loader:

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

################################################################
# configs
################################################################

resolution = 20
num_cut = 10
bjorn = True

w_size = 11
w_radius = (w_size-1)//2

modes = 8 # number of frequency modes
width = 20 # dimension of latent space
batch_size = 16
learning_rate = 0.001
epochs = 200

results_dir = osp.join('./','results')

t1 = default_timer()

################################################################
# load data
################################################################
#total_model_names = ['hollow_1','hollow_2','hollow_3','hollow_4','hollow_5','townhouse_2','townhouse_3','townhouse_5','townhouse_6','townhouse_7']
total_model_names = ['hollow_1','hollow_2','hollow_3']
for valid_model_name in total_model_names:
    print(valid_model_name)
    s = set([valid_model_name])
    model_names = [x for x in total_model_names if x not in s]
    print(model_names)
    a = []
    u = []
    path = 'local_FNO'+'_ep' + str(epochs) + '_m' + str(modes) \
        + '_w' + str(width) + '_window_sz' + str(w_size) + '_' + valid_model_name
    path_model = 'model/'+path
    
    for model_name in model_names:
        ml_data_dir = osp.join(data_dir,"ml_data",model_name)
        data_path = osp.join(ml_data_dir,model_name+'_global.pk')
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
    ml_data_dir = osp.join(data_dir,"ml_data",valid_model_name)
    data_path = osp.join(ml_data_dir,valid_model_name+'_global.pk')
    valid_data = pickle.load(open( data_path, "rb" ))
    valid_a = torch.from_numpy(valid_data["a"])
    valid_u = torch.from_numpy(valid_data["u"])
    valid_a[:,:,:,:,:-1] = a_normalizer.encode(valid_a[:,:,:,:,:-1])
    valid_u = y_normalizer.encode(valid_u)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_a, valid_u), batch_size=batch_size, shuffle=False, drop_last = True)


    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda:1')
    cpu_device = torch.device('cpu')

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO3d(modes, modes, modes, width)
    model = model.to(device)
    print(f'parameter number: {count_params(model)}')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    cur_ep = 0
    #if os.path.isfile(path_model):
    # checkpoint = torch.load(path_model)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #cur_ep = checkpoint['epoch']

    myloss = LpLoss(size_average=False)
    y_normalizer = y_normalizer.to(device)
    train_logger = Logger(
        osp.join(results_dir, valid_model_name+'_train.log'),
        ['ep', 'train_mse','train_l2','train_r2','train_mse_center','test_mse','test_l2','test_r2','test_mse_center']
    )
    
    world_res = resolution*2
    ni,mi,li = np.indices((world_res,world_res,world_res))
    ni = torch.from_numpy(ni)
    mi = torch.from_numpy(mi)
    li = torch.from_numpy(li)

    sample_region = np.array([[10,15],
                              [10,15],
                              [10,15]],dtype=np.int32)

    cur_test_r2 = 0
    for ep in range(cur_ep+1,epochs):
        ### randomly select a window
        train_mse,train_l2,train_r2,train_mse_center = train()
        print(f'ep: {ep},train mse: {train_mse:.6f},train L2: {train_l2:.6f},train r2: {train_r2:.6f},train mse center: {train_mse_center:.6f}')
        
        test_mse,test_l2,test_r2,test_mse_center = test(test_loader)
        print(f'ep: {ep},test mse: {test_mse:.6f},test L2: {test_l2:.6f},test r2: {test_r2:.6f},test mse center: {test_mse_center:.6f}')
        train_logger.log({
            'ep': ep,             
            'train_mse': train_mse,
            'train_l2': train_l2,
            'train_r2': train_r2,
            'train_mse_center':train_mse_center,
            'test_mse': test_mse,
            'test_l2': test_l2,
            'test_r2': test_r2,
            'test_mse_center':test_mse_center,
        })
        if test_r2 > cur_test_r2:
            cur_test_r2 = test_r2
            torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'test_r2': test_r2,
                        'test_mse': test_mse,
                        'test_l2':test_l2,
                        }, path_model)
        if ep % 10 == 0:
            valid_mse,valid_l2,valid_r2,valid_mse_center = test(valid_loader)
            print(f'valid mse: {valid_mse:.6f},valid L2: {valid_l2:.6f},valid r2: {valid_r2:.6f},valid mse center: {valid_mse_center:.6f}')
     


