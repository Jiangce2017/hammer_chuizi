# prepare data file for Bjorn

# node list
# element list
# activation time

import trimesh
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os.path as osp
import os
import glob
from pathlib import Path
import argparse
from hammer.utilities.plotting import plot_element_adding, plot_toolpath,plot_toolpath_with_voxels

# prepare data file for Bjorn

# node list
# element list
# activation time
data_dir = os.path.join(Path.home(), 'data','hammer') 

def prepare_bjorn_data(args):
    problem_name = args.problem_name
    femfile_dir = osp.join(data_dir,"meshes",problem_name)
    files = glob.glob(osp.join(femfile_dir, f'*'))
    for ffi,f in enumerate(files):
        fem_file = pickle.load(open(f, "rb" ) ) 
    
        bjorn_dir = os.path.join(data_dir,"bjorn",problem_name,Path(f).stem)
        os.makedirs(bjorn_dir, exist_ok=True)
        bjorn_files = glob.glob(os.path.join(bjorn_dir, f'*'))
        for ff in bjorn_files:
            os.remove(ff)
    
        vertices, cells, toolpath, sampled_deposits = fem_file["extend_vertices"],fem_file["hexahedra"], fem_file["toolpath"],\
            fem_file['sampled_deposits']
        cells = cells + 1 ### for matlab 
    
        toolpath[:, 1:4] = toolpath[:, 1:4]
        
        node_ind = np.expand_dims(np.arange(vertices.shape[0])+1, axis=1)   
        nodes_list = np.concatenate((node_ind, vertices),axis=1)   
        cell_ind = np.expand_dims(np.arange(cells.shape[0])+1, axis=1)  
        cell_list = np.concatenate((cell_ind, cells),axis=1)
        cell_list = cell_list.astype(np.int32)
        dt = fem_file["dt"]
        dx = fem_file["dx"]
        print(dt,dx,dx/dt)
        activation_time = -1*np.ones((cells.shape[0],1),dtype=np.float64)
        num_base = cells.shape[0]-toolpath.shape[0]
        activation_time[num_base,0] = 0
        for i_ele in range(1,toolpath.shape[0]):
            direction = toolpath[i_ele, 1:4] - toolpath[i_ele - 1 , 1:4]
            d = np.linalg.norm(direction)
            n_step = np.round(d/dx)
            activation_time[i_ele+num_base,0] = dt*n_step + activation_time[i_ele+num_base-1,0]
        
        sampled_timesteps = np.zeros(cells.shape[0],dtype=np.int32)
        sampled_timesteps[sampled_deposits+num_base] = 1        
        
        np.savetxt(osp.join(bjorn_dir, problem_name+"_nodes.txt"), nodes_list, delimiter=',')
        np.savetxt(osp.join(bjorn_dir, problem_name+"_cells.txt"), cell_list, fmt='%d', delimiter=',')
        np.savetxt(osp.join(bjorn_dir, problem_name+"_activation_time.txt"), activation_time, delimiter=',')
        np.savetxt(osp.join(bjorn_dir, problem_name+"_sampled_timesteps.txt"), sampled_timesteps,fmt='%d', delimiter=',')
        print("Finished {} files".format(ffi+1))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_name", type=str, required=True)
    args = parser.parse_args() 
    prepare_bjorn_data(args)