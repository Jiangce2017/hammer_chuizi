import matplotlib.pyplot as plt
import pickle
import numpy as np
from plotting import plot_data
data_50 = pickle.load(open('results_cut10_plot_50.pk', "rb" ))
data_20 = pickle.load(open('results_cut10.pk', "rb" ))
data_valid = pickle.load(open('results_cut10townhouse_2.pk', "rb" ))

# PLOT 50 EPOCH DATA
file_name_50 = "50_epochs"
# Use when data is not in numpy array
a_50 = np.concatenate(data_50['a'], axis=0)
true_u_50 = np.concatenate(data_50['true_u'],axis=0)

# Remove Last Dimension
a_50 = a_50[:, :, :, :, 0]  # input
true_u_50 = true_u_50[:, :, :, :, 0]
pred_u_50 = data_50['pred_u'][:, :, :, :, 0]



# PLOT 20 EPOCH DATA
file_name_20 = "20_epochs"
# Use when data is in a numpy array
a_20 = data_20['a']
true_u_20 = data_20['true_u']

# Remove Last Dimension
# activation = data['a'][:,:,:,:,6]
a_20 = a_20[:, :, :, :, 0]  # input

true_u_20 = true_u_20[:, :, :, :, 0]
pred_u_20 = data_20['pred_u'][:, :, :, :, 0]


# Get window information and time steps from 20 epoch data
window_info = data_20['window_info']
time_steps = window_info[:,0]
window_numbers = window_info[:,1]


# PLOT VALIDATION DATA
file_name_valid = "worst_validation_error"

# true_u_valid = np.concatenate(data_valid['true_u'],axis=0)
true_u_valid = data_valid['true_u'][:, :, :, :, 0]  # Remove Last Dimension
pred_u_valid = data_valid['pred_u'][:, :, :, :, 0]  # Remove Last Dimension
# Get window information
window_info_valid = data_valid['window_info']
time_steps_valid = window_info_valid[:,0]
window_numbers_valid = window_info_valid[:,1]


# Establish plotting conditions
number_samples = 10
columns = 4

random_samples = sorted(np.random.randint(0,len(true_u_20), number_samples))  # Generates 10 ordered samples

fontsize = number_samples * 7

# Plot/Save the figures
fig1 = plt.figure(figsize=(16*columns, number_samples*10))
plot_data(true_u_20, pred_u_20, window_numbers, time_steps, random_samples, file_name_20, fig1, fontsize)

fig2 = plt.figure(figsize=(16*columns, number_samples*10))
plot_data(true_u_50, pred_u_50, window_numbers, time_steps, random_samples, file_name_50, fig2, fontsize)


fig3 = fig2 = plt.figure(figsize=(16*columns, number_samples*10))
plot_data(true_u_valid, pred_u_valid, window_numbers_valid, time_steps_valid, random_samples, file_name_valid, fig3, fontsize, worst_error=True)
fig4 = plt.figure(figsize=(16*columns, number_samples*10))
plot_data(true_u_valid, pred_u_valid, window_numbers_valid, time_steps_valid, random_samples, "validation", fig4, fontsize)

# index = 9167
# error = abs(true_u[index] - pred_u[index])
# u_error = np.where(true_u[index] == 0, 0, np.divide(abs(error),abs(true_u[index])) * 100)
# matplot_voxels(error, fontsize=fontsize, cmap='Reds')
# plt.show()
#
# matplot_voxels(u_error, fontsize=fontsize, cmap='Reds')
# plt.show()