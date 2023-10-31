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
#from hammer import 
#from models import get_window,FNO3d, r2loss
from hammer import FNO3d, r2loss, CNN

import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import csv
torch.manual_seed(1234)
np.random.seed(1234)

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

    for x_train_batch, y_train_batch in train_loader:
    
        x_train, y_train = get_window(sample_region,world_res,ni,mi,li,w_radius,w_size,batch_size,x_train_batch,y_train_batch)
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        rho = x_train[:,:,:,:,[-1]]
        optimizer.zero_grad()
        pred_train = model(x_train).view(batch_size, w_size, w_size, w_size,1)
        pred_train *= rho
        #y_train *= rho
        
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

def chunk_domain(x, chunk_size,start_idx=0,end_idx=[33,33,33],domain_dim=3):
    if domain_dim == 3:
        x = x[:,start_idx:end_idx[0],start_idx:end_idx[1],start_idx:end_idx[2],:]
        x = torch.split(x,chunk_size,dim=1)
        x = torch.cat(x,dim=0)
        x = torch.split(x,chunk_size,dim=2)
        x = torch.cat(x,dim=0)
        x = torch.split(x,chunk_size,dim=3)
        x = torch.cat(x,dim=0)
    elif domain_dim == 2:
        x = x[:,start_idx:end_idx[0],start_idx:end_idx[1],:]
        x = torch.split(x,chunk_size,dim=1)
        x = torch.cat(x,dim=0)
        x = torch.split(x,chunk_size,dim=2)
        x = torch.cat(x,dim=0)
    return x

def patch_window(x,bz,chunk_number,domain_dim=3):
    if domain_dim == 3:
        x = torch.split(x,int(bz*chunk_number[0]*chunk_number[1]),dim=0)
        x = torch.cat(x,dim=3)
        
        x = torch.split(x,int(bz*chunk_number[0]),dim=0)
        x = torch.cat(x,dim=2)
        x = torch.split(x,bz*1,dim=0)
        x = torch.cat(x,dim=1)
    elif domain_dim == 2:        
        x = torch.split(x,int(bz*chunk_number[0]),dim=0)
        x = torch.cat(x,dim=2)
        x = torch.split(x,bz*1,dim=0)
        x = torch.cat(x,dim=1)
    
    return x

def train_split(chunk_number,num_grid_move_train,w_size):
    model.train()
    train_mse = 0.0
    train_l2 = 0.0
    train_r2 = 0.0
    train_mse_center = 0
    num_sample = 0
    
    global_train_mse = 0.0
    global_train_l2 = 0.0
    global_train_r2 = 0.0

    y_train_global_holder = torch.zeros((batch_size*np.prod(chunk_number),w_size,w_size,w_size,1)).to(device)
    pred_train_global_holder = torch.zeros((batch_size*np.prod(chunk_number),w_size,w_size,w_size,1)).to(device)
    for x_train_batch, y_train_batch in train_loader:
        for start_idx in range(num_grid_move_train):
            #print(y_train_global.shape)
            x_train = chunk_domain(x_train_batch,w_size,start_idx,start_idx+w_size*chunk_number)
            y_train = chunk_domain(y_train_batch,w_size,start_idx,start_idx+w_size*chunk_number)
            ### filter out void windows
            
            solid_windows = torch.sum(x_train[:,:,:,:,-1],dim=[1,2,3]) > 0.1*w_size**domain_dim
            x_train = x_train[solid_windows]
            y_train = y_train[solid_windows]

            num_window = x_train.shape[0]
            if num_window > 0:
                
                x_train, y_train = x_train.to(device), y_train.to(device)
                
                rho = x_train[:,:,:,:,[-1]]
                optimizer.zero_grad()
                pred_train = model(x_train).view(num_window, w_size, w_size, w_size,1)
                pred_train *= rho
                y_train *= rho
                
                
                l2 = myloss(pred_train.reshape(num_window, -1), y_train.reshape(num_window, -1))
                
                l2.backward()
                optimizer.step()
                scheduler.step()

                mse = F.mse_loss(pred_train, y_train, reduction='mean')
                r2 = r2loss(pred_train.reshape(num_window, -1), y_train.reshape(num_window, -1))
                pred_train_center = pred_train[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
                y_train_center = y_train[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
                mse_center = F.mse_loss(pred_train_center, y_train_center, reduction='mean')

                pred_train_global_holder[solid_windows] = pred_train
                y_train_global_holder[solid_windows] = y_train
                pred_train_global = patch_window(pred_train_global_holder,batch_size,chunk_number)
                y_train_global = patch_window(y_train_global_holder,batch_size,chunk_number)
                global_mse = F.mse_loss(pred_train_global, y_train_global, reduction='mean')
                global_l2 = myloss(pred_train_global.reshape(batch_size, -1), y_train_global.reshape(batch_size, -1))
                global_r2 = r2loss(pred_train_global.reshape(batch_size, -1), y_train_global.reshape(batch_size, -1))

                train_mse += mse.item()
                train_l2 += l2.item()
                train_r2 += r2.item()
                train_mse_center += mse_center.item()
                
                global_train_mse += global_mse.item()
                global_train_l2 += global_l2.item()
                global_train_r2 += global_r2.item()
                num_sample += 1
        
    train_mse /= num_sample
    train_l2 /= num_sample
    train_r2 /= num_sample
    train_mse_center /= num_sample
    global_train_mse /= num_sample
    global_train_l2 /= num_sample
    global_train_r2 /= num_sample
    return train_mse,train_l2,train_r2,train_mse_center,global_train_mse,global_train_l2,global_train_r2

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

def test_split(test_loader,chunk_number,num_grid_move_test,w_size):
    model.eval()              
    test_mse = 0.0
    test_l2 = 0.0
    test_r2 = 0.0     
    test_mse_center = 0.0
    num_sample = 0

    global_test_mse = 0.0
    global_test_l2 = 0.0
    global_test_r2 = 0.0
    num_global_sample = 0

    with torch.no_grad():
        for x_test_branch, y_test_branch in test_loader:
            x_test_branch, y_test_branch = x_test_branch.to(device), y_test_branch.to(device)
            pred_test_branch_holder = torch.zeros(y_test_branch.shape).to(device)
            y_test_branch_holder = torch.zeros(y_test_branch.shape).to(device)
            rho_branch_holder = torch.zeros(y_test_branch.shape).to(device)
            for start_idx in range(num_grid_move_test):
                end_idx = start_idx+w_size*chunk_number
                x_test = chunk_domain(x_test_branch,w_size,start_idx,end_idx)
                y_test = chunk_domain(y_test_branch,w_size,start_idx,end_idx)
                ### filter out void windows
                solid_windows = torch.sum(x_test[:,:,:,:,-1],dim=[1,2,3]) > 0.1*w_size**domain_dim
                num_window = x_test.shape[0]
                num_solid_window = x_test[solid_windows].shape[0]
                
                rho = x_test[:,:,:,:,[-1]]

                pred_test = model(x_test).view(num_window, w_size, w_size, w_size,1)     
                
                pred_test *= rho
                y_test *= rho
                
                test_mse += F.mse_loss(pred_test, y_test, reduction='mean').item()             
                
                test_l2 += myloss(pred_test[solid_windows].view(num_solid_window, -1), y_test[solid_windows].view(num_solid_window, -1)).item()
                test_r2 += r2loss(pred_test[solid_windows].view(num_solid_window, -1), y_test[solid_windows].view(num_solid_window, -1)).item()

                pred_test_center = pred_test[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
                y_test_center = y_test[:,w_radius-1:w_radius+1,w_radius-1:w_radius+1,w_radius-1:w_radius+1,:]
                mse_center = F.mse_loss(pred_test_center, y_test_center, reduction='mean')

                test_mse_center += mse_center.item()
                num_sample += 1
                pred_test_branch_holder[:,start_idx:end_idx[0],start_idx:end_idx[1],start_idx:end_idx[2],:] += patch_window(pred_test,batch_size,chunk_number)
                y_test_branch_holder[:,start_idx:end_idx[0],start_idx:end_idx[1],start_idx:end_idx[2],:]  += patch_window(y_test,batch_size,chunk_number)
                rho_branch_holder[:,start_idx:end_idx[0],start_idx:end_idx[1],start_idx:end_idx[2],:]  += patch_window(x_test[:,:,:,:,[-1]],batch_size,chunk_number)
            
            
            pred_test_branch_holder[rho_branch_holder > 0] /= rho_branch_holder[rho_branch_holder > 0]
            y_test_branch_holder[rho_branch_holder > 0] /= rho_branch_holder[rho_branch_holder > 0]
            zero_mask = y_test_branch_holder == 0
            rho = x_test_branch[:,:,:,:,[-1]]
            pred_test_branch_holder *= rho
            y_test_branch *= rho
            y_test_branch[zero_mask] = 0
            y_test_branch_holder *= rho
            global_mse = F.mse_loss(y_test_branch, pred_test_branch_holder, reduction='mean')
            global_l2 = myloss(y_test_branch.view(batch_size, -1), pred_test_branch_holder.view(batch_size, -1))
            global_r2 = r2loss(pred_test_branch_holder.reshape(batch_size, -1), y_test_branch.reshape(batch_size, -1))
            
            global_test_mse += global_mse.item()
            global_test_l2 += global_l2.item()
            global_test_r2 += global_r2.item()
            num_global_sample += 1
    test_mse /= num_sample
    test_l2 /= num_sample
    test_r2 /= num_sample
    test_mse_center /= num_sample
    
    global_test_mse /= num_global_sample
    global_test_l2 /= num_global_sample
    global_test_r2 /= num_global_sample
    return test_mse,test_l2,test_r2,test_mse_center,global_test_mse,global_test_l2,global_test_r2

################################################################
# configs
################################################################

resolution = np.array([20,20,20],dtype=int)
bjorn = False
domain_dim = 3
w_size = 11
w_radius = (w_size-1)//2
world_res = resolution*2
num_grid_move_train = w_radius
num_grid_move_test = w_radius
chunk_number_train = (world_res-num_grid_move_train) // w_size
chunk_number_test = (world_res-num_grid_move_test) // w_size


modes = 8 # number of frequency modes 8 for 11 w_size
width = 20 # dimension of latent space
batch_size = 8
learning_rate = 0.0001
epochs = 50

results_dir = osp.join('./','results')

t1 = default_timer()

################################################################
# load data
################################################################
total_model_names = ['hollow_1','hollow_2','hollow_3','hollow_4','hollow_5','townhouse_2','townhouse_3','townhouse_5','townhouse_6','townhouse_7',]
#total_model_names = ['hollow_1','hollow_2',]
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
    print(a.shape)
    print(u.shape)
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
    #train_u = y_normalizer.encode(train_u)
    #test_u = y_normalizer.encode(test_u)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, drop_last = True)

    t2 = default_timer()


    #### valid model 
    ml_data_dir = osp.join(data_dir,"ml_data",valid_model_name)
    valid_data_path = osp.join(ml_data_dir,valid_model_name+'_global.pk')
    valid_data = pickle.load(open( valid_data_path, "rb" ))
    valid_a = torch.from_numpy(valid_data["a"])
    valid_u = torch.from_numpy(valid_data["u"])
    valid_a[:,:,:,:,:-1] = a_normalizer.encode(valid_a[:,:,:,:,:-1])
    #valid_u = y_normalizer.encode(valid_u)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_a, valid_u), batch_size=batch_size, shuffle=False, drop_last = True)


    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda:0')
    cpu_device = torch.device('cpu')

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO3d(modes, modes, modes, width)
    #model = CNN(modes, modes, modes, 128)
    model = model.to(device)
    print(f'parameter number: {count_params(model)}')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    cur_ep = 0
    # if os.path.isfile(path_model):
    #     checkpoint = torch.load(path_model)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     cur_ep = checkpoint['epoch']

    myloss = LpLoss(size_average=False)
    y_normalizer = y_normalizer.to(device)
    train_logger = Logger(
        osp.join(results_dir, valid_model_name+'_train.log'),
        ['ep', 'train_mse','train_l2','train_r2','train_mse_center','test_mse','test_l2','test_r2','test_mse_center']
    )
    
    world_res = resolution*2
    ni,mi,li = np.indices((world_res[0],world_res[1],world_res[2]))
    ni = torch.from_numpy(ni) 
    mi = torch.from_numpy(mi)
    li = torch.from_numpy(li)

    sample_region = np.array([[10,30],
                              [10,30],
                              [10,15]],dtype=np.int32)

    cur_test_r2 = 0
    for ep in range(cur_ep+1,epochs):
        ### randomly select a window
        train_mse,train_l2,train_r2,train_mse_center,global_train_mse,global_train_l2,global_train_r2 = train_split(chunk_number_train,num_grid_move_train,w_size)
        print(f'ep: {ep},train mse: {train_mse:.6f},train L2: {train_l2:.6f},train r2: {train_r2:.6f},train mse center: {train_mse_center:.6f},global train mse: {global_train_mse:.6f}, global train l2: {global_train_l2:.6f},global train r2: {global_train_r2:.6f}')
        
        test_mse,test_l2,test_r2,test_mse_center,global_test_mse,global_test_l2,global_test_r2 = test_split(test_loader,chunk_number_test,num_grid_move_test,w_size)
        print(f'ep: {ep},test mse: {test_mse:.6f},test L2: {test_l2:.6f},test r2: {test_r2:.6f},test mse center: {test_mse_center:.6f},global test mse: {global_test_mse:.6f}, global test l2: {global_test_l2:.6f},global test r2: {global_test_r2:.6f}')
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
        if ep % 1 == 0:
            valid_mse,valid_l2,valid_r2,valid_mse_center,global_valid_mse,global_valid_l2,global_valid_r2 = test_split(valid_loader,chunk_number_test,num_grid_move_test,w_size)
            print(f'valid mse: {valid_mse:.6f},valid L2: {valid_l2:.6f},valid r2: {valid_r2:.6f},valid mse center: {valid_mse_center:.6f},global valid mse: {global_valid_mse:.6f}, global valid l2: {global_valid_l2:.6f},global valid r2: {global_valid_r2:.6f}')
     


