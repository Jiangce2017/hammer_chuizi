import meshio
import os.path as osp
import os
from pathlib import Path
import trimesh
import numpy as np
import pickle
from numpy import linalg as LA

from hammer import GeoReader
from hammer import AMMesh, AMGraph, AMVoxel
import torch

data_dir = os.path.join(Path.home(), 'data','hammer') 

def preprocess_graph_data(model_name,bjorn=True):
    problem_name = "small_10_base_20"
    femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
    if bjorn:
        vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    else:    
        vtk_dir = osp.join(data_dir,"vtk",problem_name, model_name)


    fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 

    voxel_inds, deposit_sequence, toolpath = fem_file["voxel_inds"], fem_file["deposit_sequence"], fem_file["toolpath"]
    dx = fem_file["dx"]
    nx = fem_file["nx"]
    Add_base = fem_file["Add_base"]
    path_dx = fem_file["dx"]
    if bjorn:
        whole_cells = fem_file["hexahedra"]
        num_base = whole_cells.shape[0]-toolpath.shape[0]

    #base_depth = fem_file["base_depth"]
    base_depth = 50e-3
    ratio = np.abs(np.min(fem_file["vertices"][:,2]))/base_depth

    deposit_pairs = fem_file["deposit_pairs"]

    #for i_sample in range(10,deposit_pairs.shape[0]):
    for i_sample in range(10,20):
        ### load vtu files
        if bjorn:
            i_time_step = num_base + deposit_pairs[i_sample,0] + 1
            vtk_path_00 = osp.join(vtk_dir, f"T{(i_time_step):07}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"T{(i_time_step+1):07d}.vtu")
        else:
            i_time_step = deposit_pairs[i_sample,0]
            vtk_path_00 = osp.join(vtk_dir, f"u_{(i_time_step):05d}_active_{0:05d}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"u_{(i_time_step+1):05d}_active_{0:05d}.vtu")
        print(i_time_step)
        if os.path.isfile(vtk_path_00) and os.path.isfile(vtk_path_01):
            mesh0 = AMMesh(path=vtk_path_00)
            mesh1 = AMMesh(path=vtk_path_01)
            graph0 = mesh0.to("Graph")
            graph1 = mesh1.to("Graph")
            node_features0, edge_index = graph0.torch_data()
            node_features1, _ = graph1.torch_data()
            
            node_features1 = node_features1[:-1] 
            assert node_features0.shape[0] == node_features1.shape[0]
            #print(node_features0.shape[0])
            #######################
            # i_deposit = deposit_pairs[i_sample,0]
            # lag = 1
            # i_deposit -= lag
            #laser_center = np.array([toolpath[i_deposit,1], toolpath[i_deposit,2], toolpath[i_deposit,3]])
            laser_center = node_features1[-1,:3]
            voxel0 = mesh0.to("Voxel")
            heat_info = laser_center-voxel0.center_coords ###  input laser position      
            print(heat_info)
            input_T = node_features0[:,[-1]]
            output_T = node_features1[:,[-1]]
            pos = node_features0[:,:3]
            
            
            
            
            
            
if __name__ == '__main__':
    model_name = "hollow_5"            
    preprocess_graph_data(model_name)
          
                
    