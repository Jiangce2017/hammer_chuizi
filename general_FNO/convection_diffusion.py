import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from FNO_2d import DomainPartitioning2d
import wandb


def plot_prediction(window_size, y, y_pred, epoch, batch_idx, folder):
    xx, yy = np.meshgrid(np.linspace(0, 1, window_size), np.linspace(0, 1, window_size))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].contourf(xx, yy, y.cpu().detach().numpy().reshape(window_size, window_size), levels=100)
    axs[0].set_title('Ground truth')
    axs[1].contourf(xx, yy, y_pred.cpu().detach().numpy().reshape(window_size, window_size), levels=100)
    axs[1].set_title('Prediction')
    axs[2].contourf(xx, yy, np.abs(y.cpu().detach().numpy().reshape(window_size, window_size) - y_pred.cpu().detach().numpy().reshape(window_size, window_size)), levels=100)
    axs[2].set_title('Absolute difference')
    wandb.log({folder + '/Ground truth': wandb.Image(axs[0])})
    wandb.log({folder + '/Prediction': wandb.Image(axs[1])})
    wandb.log({folder + '/Absolute difference': wandb.Image(axs[2])})
    plt.close()

class ConvectionDiffusion:
    def __init__(self, c, d, wave_direction, wave_frequency, domain_size, resolution, num_time_steps, dt, seed=0):
        """
        generate one 2D simulation of the convection-diffusion equation
        :param c: convection coefficient
        :param d: diffusion coefficient
        :param domain_size: size of the domain (square domain)
        :param num_time_steps: number of time steps to simulate
        :param dt: time step size
        :param seed: random seed
        """
        self.c = c
        self.d = d
        self.wave_direction = wave_direction
        self.wave_frequency = wave_frequency
        self.domain_size = domain_size
        self.resolution = resolution
        # self.num_samples = num_samples
        self.num_time_steps = num_time_steps
        self.dt = dt
        self.seed = seed

    def _compute_exact_solution_at_t(self, time_step):
        """
        compute the exact solution of the convection-diffusion equation
        :return: exact solution
        """
        # compute the exact solution
        x = np.linspace(0, self.domain_size, self.resolution)
        y = np.linspace(0, self.domain_size, self.resolution)
        xx, yy = np.meshgrid(x, y)
        u = np.exp(-2 * self.d * np.pi ** 2 * time_step * self.dt) * np.sin(self.wave_frequency * np.pi / (self.c) * (xx - self.c * time_step * self.dt * np.cos(self.wave_direction)))
        return np.expand_dims(u, axis=2)
        # return u

    def compute(self):
        solution = []
        for t in range(self.num_time_steps):
            solution.append(self._compute_exact_solution_at_t(t))

        return np.array(solution, dtype=np.float32)


class ConvectionDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, frequency, domain_size, resolution, num_time_steps, dt, num_samples, seed=0):
        """
        generate a dataset of 2D simulations of the convection-diffusion equation
        :param domain_size: size of the domain (square domain)
        :param num_time_steps: number of time steps to simulate
        :param dt: time step size
        :param num_samples: number of samples to generate
        :param seed: random seed
        """
        self.domain_size = domain_size
        self.resolution = resolution
        self.num_time_steps = num_time_steps
        self.dt = dt
        self.num_samples = num_samples
        self.seed = seed
        self.frequency = frequency

        self._set_up()

    def _set_up(self):
        """
        set up the dataset
        """
        # set up the random seed
        np.random.seed(self.seed)
        solutions = []
        for i in range(self.num_samples):
            # c = np.random.uniform(0.1, 1)
            c = 0.5
            # d = np.random.uniform(0.1, 1)
            d = 0.001
            direction = np.random.uniform(0, np.pi)
            # frequency = np.random.randint(1, 5)
            frequency = self.frequency
            convection_diffusion = ConvectionDiffusion(c, d, direction, frequency, self.domain_size, self.resolution, self.num_time_steps, self.dt)
            solution = convection_diffusion.compute()
            solutions.append(solution)
        solutions = np.array(solutions)

        # organize the solution into input, label pairs, with squential time step
        self.input = []
        self.label = []
        for i in range(self.num_samples):
            for t in range(self.num_time_steps-1):
                self.input.append(solutions[i, t])
                self.label.append(solutions[i, t + 1])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input[idx], self.label[idx]


def run_experiment(config: dict):
    wandb.init(project='Domain_partition_2D', config=config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # read parameters from config
    domain_size = config['domain_size']
    resolution = config['resolution']
    num_time_steps = config['num_time_steps']
    dt = config['dt']
    num_samples = config['num_samples']
    seed = config['seed']
    modes = config['modes']
    width = config['width']
    window_size = config['window_size']
    num_iterations = config['num_iterations']
    data_frequency = config['data_frequency']

    # initialize model
    dataset = ConvectionDiffusionDataset(data_frequency, domain_size, resolution, num_time_steps, dt, num_samples, seed)
    model = DomainPartitioning2d(modes, modes, width, window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(0.5 * len(val_dataset)), len(val_dataset) - int(0.5 * len(val_dataset))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # training loop
    for epoch in range(num_iterations):
        model.train()
        train_mse_loss = 0
        train_r2_accuracy = 0
        reconstructed_r2_accuracy = 0
        for x, y in train_loader:
            # print(x.shape)
            # print(y.shape)
            sub_x_list = model.get_partition_domain(x, mode='train')
            sub_y_list = model.get_partition_domain(y, mode='test')
            pred_list = []
            for sub_x, sub_y in zip(sub_x_list, sub_y_list):
                sub_x = sub_x.to(device)
                sub_y = sub_y.to(device)
                optimizer.zero_grad()
                pred = model(sub_x)
                pred_list.append(pred)
                loss = F.mse_loss(pred, sub_y)
                train_r2_accuracy += r2_score(sub_y.detach().cpu().numpy().reshape(128, -1), pred.detach().cpu().numpy().reshape(128, -1))
                loss.backward()
                optimizer.step()
                train_mse_loss += loss.item()
            
            pred_y = model.reconstruct_from_partitions(y, pred_list)
            print(pred_y.shape)
            reconstructed_r2_accuracy += r2_score(y.detach().cpu().numpy().reshape(128, -1), pred_y.detach().cpu().numpy().reshape(128, -1))

        train_mse_loss /= (len(train_loader) * len(sub_x_list))
        train_r2_accuracy /= (len(train_loader) * len(sub_x_list))
        reconstructed_r2_accuracy /= len(train_loader)

        wandb.log({'train_mse_loss': train_mse_loss, 'train_r2_accuracy': train_r2_accuracy, 'reconstructed_r2_accuracy': reconstructed_r2_accuracy})

        if epoch % 40 == 0:
            model.eval()
            val_l2_loss = 0
            val_r2_accuracy = 0
            for x, y in val_loader:
                sub_x_list = model.get_partition_domain(x, mode='train')
                sub_y_list = model.get_partition_domain(y, mode='test')
                pred_list = []
                # print(x.shape)
                # print(y.shape)
                for sub_x, sub_y in zip(sub_x_list, sub_y_list):
                    sub_x = sub_x.to(device)
                    sub_y = sub_y.to(device)
                    pred = model(sub_x)
                    pred_list.append(pred)
                    loss = F.mse_loss(pred, sub_y)
                    val_r2_accuracy += r2_score(sub_y.detach().cpu().numpy().reshape(sub_y.shape[0], -1), pred.detach().cpu().numpy().reshape(pred.shape[0], -1))
                    # val_r2_accuracy += r2_score(sub_y.detach().cpu().numpy(), pred.detach().cpu().numpy())
                    val_l2_loss += loss.item()

                pred_y = model.reconstruct_from_partitions(y, pred_list)
                reconstructed_r2_accuracy += r2_score(y.detach().cpu().numpy().reshape(sub_y.shape[0], -1), pred_y.detach().cpu().numpy().reshape(pred_y.shape[0], -1))
            val_l2_loss /= (len(val_loader) * len(sub_x_list))
            val_r2_accuracy /= (len(val_loader) * len(sub_x_list))
            reconstructed_r2_accuracy /= len(val_loader)

            wandb.log({'val_l2_loss': val_l2_loss, 'val_r2_accuracy': val_r2_accuracy, 'reconstructed_r2_accuracy': reconstructed_r2_accuracy})

            plot_prediction(resolution, y[0], pred_y[0], epoch, 0, 'results')
            plot_prediction(window_size, sub_y[0], pred[0], epoch, 0, 'results_subdomain')
        
        scheduler.step()

    # test
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_l2_loss = 0
    test_r2_accuracy = 0
    for x, y in test_loader:
        sub_x_list = model.get_partition_domain(x)
        sub_y_list = model.get_partition_domain(y)
        pred_list = []
        for sub_x, sub_y in zip(sub_x_list, sub_y_list):
            sub_x = sub_x.to(device)
            sub_y = sub_y.to(device)
            pred = model(sub_x)
            pred_list.append(pred)
            loss = F.mse_loss(pred, sub_y)
            test_r2_accuracy += r2_score(sub_y.detach().cpu().numpy().reshape(sub_y.shape[0], -1), pred.detach().cpu().numpy().reshape(pred.shape[0], -1))
            test_l2_loss += loss.item()

        pred_y = model.reconstruct_from_partitions(y, pred_list)
        reconstructed_r2_accuracy += r2_score(y.detach().cpu().numpy().reshape(sub_y.shape[0], -1), pred_y.detach().cpu().numpy().reshape(pred_y.shape[0], -1))
    test_l2_loss /= (len(test_loader) * len(sub_x_list))
    test_r2_accuracy /= (len(test_loader) * len(sub_x_list))
    reconstructed_r2_accuracy /= len(test_loader)

    wandb.log({'test_l2_loss': test_l2_loss, 'test_r2_accuracy': test_r2_accuracy, 'reconstructed_r2_accuracy': reconstructed_r2_accuracy})
    wandb.finish()


if __name__ == '__main__':
    # wandb.init(project='Domain_partition_2D')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # set up the dataset
    domain_size = 1
    resolution = 48
    num_time_steps = 500
    dt = 0.1
    num_samples = 2000
    seed = 0
    # modes = [2, 4, 6, 8]
    mode = 8

    width = 20
    data_frequency = [0.1, 0.5, 1, 2, 5, 10]
    window_size = [6, 8, 10, 12, 14, 16, 18, 20]
    num_iterations = 500

    for frequency in data_frequency:
        for size_ in window_size:
            # for mode in modes:
            config = dict(
                domain_size=domain_size,
                resolution=resolution,
                num_time_steps=num_time_steps,
                dt=dt,
                num_samples=num_samples,
                seed=seed,
                modes=mode,
                width=width,
                window_size=size_,
                num_iterations=num_iterations,
                data_frequency=frequency
            )

            run_experiment(config)