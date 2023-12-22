"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

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

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class DomainPartitioning2d(nn.Module):
    def __init__(self, modes1, modes2, width, subdomain_length):
        super(DomainPartitioning2d, self).__init__()
        self.sub_size = subdomain_length
        self.fno = FNO2d(modes1, modes2, width)

    # def forward(self, x):
    #     # for input data matrix x, partition the domain into num_partitions subdomains
    #     # run FNO on each of the subdomains, and then combine the results
    #     # re-partition the domain into num_partitions subdomains with a slight displacement
    #     # run FNO on each of the subdomains, and then combine the results
    #     # average the results from the two runs as the final output y
        
    #     # partition the domain into num_partitions subdomains
    #     x_list_1 = self.get_partition_domain(x)
    #     x_list_2 = self.get_partition_domain(x, displacement=self.sub_size//2)

    #     # run FNO on each of the subdomains
    #     y_list_1 = []
    #     for x_sub in x_list_1:
    #         y_sub = self.fno(x_sub)
    #         y_list_1.append(y_sub)
    #     y_list_2 = []
    #     for x_sub in x_list_2:
    #         y_sub = self.fno(x_sub)
    #         y_list_2.append(y_sub)

    #     y_1 = self.reconstruct_from_partitions(x, y_list_1, displacement=0)
    #     y_2 = self.reconstruct_from_partitions(x, y_list_2, displacement=self.sub_size//2)

    #     # average the results from the two runs as the final output y based on displacement
    #     y = y_1.clone()
    #     y[:, self.sub_size//2:, self.sub_size//2:, :] = (y_1[:, self.sub_size//2:, self.sub_size//2:, :] + y_2) / 2
    #     # y[:, :self.sub_size//2, :self.sub_size//2, :] = y_1[:, :self.sub_size//2, :self.sub_size//2, :]

    #     return y

    def forward(self, x):
        return self.fno(x)


    def get_partition_domain(self, x, displacement=0):
        # partition the domain into num_partitions subdomains of the same size
        x_list = []
        num_partitions_dim = (x.shape[1] - displacement) // self.sub_size
        # if the domain can be fully partitioned into subdomains of the same size
        if (x.shape[1] - displacement) % self.sub_size == 0:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x_list.append(x[:, i*self.sub_size:(i+1)*self.sub_size, j*self.sub_size:(j+1)*self.sub_size, :])
        # if the domain cannot be fully partitioned into subdomains of the same size
        else:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x_list.append(x[:, i*self.sub_size:(i+1)*self.sub_size, j*self.sub_size:(j+1)*self.sub_size, :])
            # add the last subdomain
            x_list.append(x[:, (x.shape[1] - self.sub_size):x.shape[1], (x.shape[2] - self.sub_size):x.shape[2], :])

        return x_list

    def reconstruct_from_partitions(self, x, x_list, displacement=0):
        # reconstruct the domain from the partitioned subdomains
        num_partitions_dim = int(np.sqrt(len(x_list)))
        x = torch.zeros_like(x[:, displacement:, displacement:, 0].unsqueeze(-1))
        # if the domain can be fully partitioned into subdomains of the same size
        if len(x_list) == num_partitions_dim**2:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x[:, i*self.sub_size:(i+1)*self.sub_size, j*self.sub_size:(j+1)*self.sub_size, :] = x_list[i*num_partitions_dim + j]
        # if the domain cannot be fully partitioned into subdomains of the same size
        else:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x[:, i*self.sub_size:(i+1)*self.sub_size, j*self.sub_size:(j+1)*self.sub_size, :] = x_list[i*num_partitions_dim + j]
            # add the last subdomain
            x[:, (x.shape[1] - self.sub_size):x.shape[1], (x.shape[2] - self.sub_size):x.shape[2], :] = x_list[-1]

        return x
