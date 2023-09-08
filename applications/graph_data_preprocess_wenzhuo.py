import meshio
import os.path as osp
import os
from pathlib import Path
import torch
# import trimesh
import numpy as np
import pickle
from torch_geometric.data import Data
from am_data_class import AMMesh, AMGraph, AMVoxel

# from hammer import GeoReader


# data_dir = os.path.join(Path.home(), 'data','hammer') 

def preprocess_graph_data(model_name):
    data_dir = 'd:/Work/research/data/hammer'
    bjorn = True
    problem_name = "small_10_base_20"
    femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
    if bjorn:
        vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    else:    
        vtk_dir = osp.join(data_dir,"vtk",problem_name, model_name)

    ml_data_dir = osp.join(data_dir,"ml_data")
    os.makedirs(ml_data_dir, exist_ok=True)

    fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 

    voxel_inds, deposit_sequence, toolpath = fem_file["voxel_inds"], fem_file["deposit_sequence"], fem_file["toolpath"]
    if bjorn:
        whole_cells = fem_file["hexahedra"]
        whole_points = fem_file["vertices"]
        num_base = whole_cells.shape[0]-toolpath.shape[0]


    #base_depth = fem_file["base_depth"]
    base_depth = 50e-3
    ratio = np.abs(np.min(fem_file["vertices"][:,2]))/base_depth

    deposit_pairs = fem_file["deposit_pairs"]
    
    for i_sample in range(10,deposit_pairs.shape[0]):
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
            i_deposit = deposit_pairs[i_sample,0]
            lag = 1
            i_deposit -= lag
            #laser_center = np.array([toolpath[i_deposit,1], toolpath[i_deposit,2], toolpath[i_deposit,3]])
            laser_center = node_features1[-1,:3]
            voxel0 = mesh0.to("Voxel")
            heat_info = torch.from_numpy(laser_center-voxel0.center_coords) ###  input laser position      
            
            T_input = node_features0[:,[-1]]
            T_output = node_features1[:,[-1]]
            pos = node_features0[:,:3]

            pairwise_data = Data(x=torch.cat([T_input, heat_info], dim=1), y=T_output, edge_index=edge_index, pos=pos)
            torch.save(pairwise_data, osp.join(ml_data_dir, f"model_{model_name}_problem_{problem_name}_{i_time_step}.pt"))
            
            ###### summary
            ###### input node attribude: input temperature, point coordinates, heat_info
            ###### output node attribude: output temperature 
            ###### build graph based on input cells 
            

          
def match_global_and_local_cell_index(whole_cells, whole_points, cells_00, points_00):
    # problem_name = "small_10_base_20"
    # femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
    # if bjorn:
    #     vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    # else:    
    #     vtk_dir = osp.join(data_dir,"vtk",problem_name, model_name)

    # ml_data_dir = osp.join(data_dir,"ml_data",model_name)
    # os.makedirs(ml_data_dir, exist_ok=True)

    # fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 

    # global cell index
    #base_depth = fem_file["base_depth"]
    base_depth = 50e-3

    # output global index of local cells by matching the coordinate of points
    cells_00_global = []
    for i in range(cells_00.shape[0]):
        for j in range(whole_cells.shape[0]):
            if np.all(whole_points[whole_cells[j]]==points_00[cells_00[i]]):
                cells_00_global.append(j)
                break
    
    cells_00_global = np.array(cells_00_global)

    # return cells_00_global
    return cells_00_global


if __name__ == "__main__":
    data_dir = "D:/Work/research/data/hammer"
    model_name = "hollow_1"
    problem_name = "small_10_base_20"
    femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
    vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    # ml_data_dir = osp.join(data_dir,"ml_data",model_name)
    # os.makedirs(ml_data_dir, exist_ok=True)

    fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 
    vtk_path_00 = osp.join(vtk_dir, f"vtk_{0:05d}.vtk")
    vtk_path_01 = osp.join(vtk_dir, f"vtk_{1:05d}.vtk")
    match_global_and_local_cell_index(fem_file, vtk_path_00, vtk_path_01, bjorn=True)