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

        self.p = nn.Linear(4, self.width) # input channel is 3: (a(x, y), x, y)
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


class DoubleConv(nn.Module):
    """
    Double Convolutional Layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class Down(nn.Module):
    """
    Downsampling Layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.down(x)
    

class Up(nn.Module):
    """
    Upsampling Layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    """
    Output Layer
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for 2D Convolutional PIML
    input: solution of the coefficient function as a 2D matrix (a(x, y))
    output: solution of the PDE (u(x, y))
    """
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        # self.down3 = Down(64, 128)
        # self.down4 = Down(128, 128)
        # self.up1 = Up(256, 64)
        # self.up2 = Up(128, 32)
        self.up3 = Up(96, 16)
        self.up4 = Up(32, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up3(x3, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        x = self.outc(x)
        return x
        

class DomainPartitioning2d(nn.Module):
    def __init__(self, modes1, modes2, width, subdomain_length):
        super(DomainPartitioning2d, self).__init__()
        self.sub_size = subdomain_length
        self.fno = FNO2d(modes1, modes2, width)
        # self.fno = UNet(1)

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
    
    def symmetric_padding(self, x, mode):
        # pad the domain symmetrically to make it divisible by sub_size
        # get pad size
        pad_size = (x.shape[1] % self.sub_size) // 2 + 1
        x = F.pad(x, (0, 0, pad_size, pad_size, pad_size, pad_size, 0, 0))
        if mode == 'train':
            # add one dimension to the tensor x, with 0 in the padded region and 1 in the original region
            x_pad_idx = torch.ones((x.shape[0], x.shape[1], x.shape[2], 1))
            x_pad_idx[:, :pad_size, :pad_size, :] = 0
            x_pad_idx[:, -pad_size:, -pad_size:, :] = 0
            x = torch.cat((x, x_pad_idx), dim=-1)
            return x, pad_size
        elif mode == 'test':    
            return x, pad_size

    def get_partition_domain(self, x, mode, displacement=0):
        # pad the domain symmetrically to make it divisible by sub_size
        x, pad_size = self.symmetric_padding(x, mode)
        # partition the domain into num_partitions subdomains of the same size
        x_list = []
        num_partitions_dim = x.shape[1] - self.sub_size + 1
        # if the domain can be fully partitioned into subdomains of the same size
        if (x.shape[1] - displacement) % self.sub_size == 0:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x_list.append(x[:, i:i+self.sub_size, j:j+self.sub_size, :])
        # if the domain cannot be fully partitioned into subdomains of the same size
        else:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x_list.append(x[:, i:i+self.sub_size, j:j+self.sub_size, :])
            # add the last subdomain
            x_list.append(x[:, (x.shape[1] - self.sub_size):x.shape[1], (x.shape[2] - self.sub_size):x.shape[2], :])

        return x_list
    

    def reconstruct_from_partitions(self, x, x_list, displacement=0):
        # reconstruct the domain from the partitioned subdomains
        num_partitions_dim = int(np.sqrt(len(x_list)))
        # print(num_partitions_dim)
        x, pad_size = self.symmetric_padding(x, mode='test')
        x = torch.zeros_like(x[:, 1:-1, 1:-1:, 0].unsqueeze(-1))
        # if the domain can be fully partitioned into subdomains of the same size
        if len(x_list) == num_partitions_dim**2:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x[:, i:i+self.sub_size-2, j:j+self.sub_size-2, :] = x_list[i*num_partitions_dim + j][:, 1:-1, 1:-1, :]
        # if the domain cannot be fully partitioned into subdomains of the same size
        else:
            for i in range(num_partitions_dim):
                for j in range(num_partitions_dim):
                    x[:, i:i+self.sub_size-2, j:j+self.sub_size-2, :] = x_list[i*num_partitions_dim + j][:, 1:-1, 1:-1, :]
            # add the last subdomain
            x[:, (x.shape[1] - self.sub_size):x.shape[1], (x.shape[2] - self.sub_size):x.shape[2], :] = x_list[-1]

        # remove the padding
        x = x[:, pad_size-1:-pad_size+1, pad_size-1:-pad_size+1, :]

        return x
