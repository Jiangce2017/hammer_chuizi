### Train ML model
"""
@author: Jiangce Chen
This file is the Small Region Fourier Neural Operator for AM thermal simulation [paper](),
"""

import meshio
import os.path as osp
import os
from pathlib import Path
import trimesh
import numpy as np
import pickle
from numpy import linalg as LA
from hammer import preprocess_data

import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

data_dir = os.path.join(Path.home(), 'data','hammer') 

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic

        self.p = nn.Linear(10, self.width)# input channel is 11: (T, Hx,Hy,Hz, H_on_off, bif1, bif2,rho, x, y, z)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        #print(x.shape)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        return x


    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


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

#TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
#TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'

#ntrain = 1000
#ntest = 200

w_size = 11

#model_name = "hollow_2"
resolution = 20
bjorn = True
#preprocess_data(w_size,model_name,resolution,bjorn)

modes = 8 # 8
width = 20 #20

batch_size = 32
learning_rate = 0.001
epochs = 500


path = 'ns_fourier_3d_N'+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
#path_image = 'image/'+path

runtime = np.zeros(2, )
t1 = default_timer()



sub = 1
S = w_size // sub
T_in = w_size
T = w_size # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;

################################################################
# load data
################################################################
model_names = ['hollow_2','hollow_3']
a = []
u = []
for model_name in model_names:
    ml_data_dir = osp.join(data_dir,"ml_data",model_name)
    data_path = osp.join(ml_data_dir,model_name+'.pk')
    #data_path = osp.join(ml_data_dir,'geo_test_cone_data.pk')
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


a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
test_u = y_normalizer.encode(test_u)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, drop_last = True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, drop_last = True)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')
#device = torch.device('cpu')

################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width)
model = model.to(device)
print(count_params(model))
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
#y_normalizer.cuda()
y_normalizer = y_normalizer.to(device)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    train_r2 = 0.0
    for x_train, y_train in train_loader:
        #x_train, y_train = x_train.cuda(), y_train.cuda()
        x_train, y_train = x_train.to(device), y_train.to(device)
        rho = x_train[:,:,:,:,[-1]]
        optimizer.zero_grad()
        pred_train = model(x_train).view(batch_size, S, S, T,1)
        pred_train *= rho
        y_train *= rho
        mse = F.mse_loss(pred_train, y_train, reduction='mean')
        #mse = mseloss(pred_train.view(batch_size, -1), y_train.view(batch_size, -1))
        # mse.backward()

        y_train = y_normalizer.decode(y_train)
        pred_train = y_normalizer.decode(pred_train)
        l2 = myloss(pred_train.view(batch_size, -1), y_train.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        scheduler.step()
        train_mse += mse.item()
        train_l2 += l2.item()
        train_r2 += r2loss(pred_train.view(batch_size, -1), y_train.view(batch_size, -1)).item()

    model.eval()
    test_mse = 0.0
    test_l2 = 0.0
    test_r2 = 0.0
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            #x_test, y_test = x_test.cuda(), y_test.cuda()
            x_test, y_test = x_test.to(device), y_test.to(device)
            rho = x_test[:,:,:,:,[-1]]
            pred_test = model(x_test).view(batch_size, S, S, T,1)
            pred_test *= rho
            y_test *= rho
            test_mse += F.mse_loss(pred_test, y_test, reduction='mean').item()
            #test_mse += mseloss(pred_test.view(batch_size, -1), y_test.view(batch_size, -1)).item()
            
            
            y_test = y_normalizer.decode(y_test)
            pred_test = y_normalizer.decode(pred_test)
            
            
            test_l2 += myloss(pred_test.view(batch_size, -1), y_test.view(batch_size, -1)).item()
            test_r2 += r2loss(pred_test.view(batch_size, -1), y_test.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= len(train_loader)
    train_r2 /= len(train_loader)
    
    test_mse /= len(test_loader)
    test_l2 /= len(test_loader)
    test_r2 /= len(test_loader)

    t2 = default_timer()
    print(f'ep: {ep}, train mse: {train_mse:.6f},train L2: {train_l2:.6f},train r2: {train_r2:.6f}, test mse: {test_mse:.6f}, test L2: {test_l2:.6f},test r2: {test_r2:.6f}')
# torch.save(model, path_model)

# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
    # for x, y in test_loader:
        # test_l2 = 0
        # x, y = x.cuda(), y.cuda()
        #  rho = x[:,:,:,:,[-1]]
        # out = model(x)
        #  out *= rho
        #  y *= rho
        # out = y_normalizer.decode(out)
        # pred[index] = out

        # test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        # #print(index, test_l2)
        # index = index + 1

#scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})




    