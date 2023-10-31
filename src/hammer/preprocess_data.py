import meshio
import os.path as osp
import os
from pathlib import Path
import trimesh
import numpy as np
import pickle
from numpy import linalg as LA

from hammer import GeoReader
from .am_data_class import AMMesh, AMGraph, AMVoxel

data_dir = os.path.join(Path.home(), 'data','hammer') 

def preprocess_data(w_size,model_name,resolution,num_cut,problem_name,femfile_dir,geo_dir,vtk_dir,ml_data_dir,bjorn=False,):
    w_radius = (w_size-1)//2

    fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 
    voxel_inds, deposit_sequence, toolpath = fem_file["voxel_inds"], fem_file["deposit_sequence"], fem_file["toolpath"]
    
    if bjorn:
        whole_cells = fem_file["hexahedra"]
        num_base = whole_cells.shape[0]-toolpath.shape[0]

    #base_depth = fem_file["base_depth"]
    base_depth = 50e-3
    ratio = np.abs(np.min(fem_file["vertices"][:,2]))/base_depth

    deposit_pairs = fem_file["deposit_pairs"]

    max_inds = np.max(voxel_inds,axis=0)
    max_coord = np.max(max_inds)

    world_res = resolution*2
    ni,mi,li = np.indices((world_res,world_res,world_res))

    ### move the model voxel into the center of world voxel
    x_shift = int((world_res - max_inds[0])//2)
    y_shift = int((world_res - max_inds[1])//2)
    z_shift = int((world_res - max_inds[2])//2)

    voxel_inds[:,0]+=x_shift
    voxel_inds[:,1]+=y_shift
    voxel_inds[:,2]+=z_shift
    ## boundary impact factor (BIF)
    dx = fem_file["dx"]
    nx = fem_file["nx"]
    Add_base = fem_file["Add_base"]
    geo_reader = GeoReader(dx,nx,Add_base)
    cad_file = osp.join(geo_dir,model_name+'.stl')
    geo_reader.load_file(cad_file)
    geo_reader.voxelize()
    bif = geo_reader.calculate_bif()
    bif_global = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],2),dtype = np.float32)
    bif_global[voxel_inds[:,0],voxel_inds[:,1],voxel_inds[:,2],:] = bif


    ## rho
    ### find base voxel
    base_plate_height = 0
    active_cell_tab = np.zeros(whole_cells.shape[0], dtype=bool)
    centroids = np.mean(fem_file["vertices"][fem_file["hexahedra"]],axis=1)
    active_cell_tab[centroids[:, 2] <= base_plate_height + dx/4 ] = True
    rho_global = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],1),dtype = np.float32)
    rho_global[voxel_inds[active_cell_tab,0],voxel_inds[active_cell_tab,1],\
               voxel_inds[active_cell_tab,2],:] = 1


    ## from indices to coordinates
    sol_global0 = np.zeros(ni.shape,dtype = np.float32)
    sol_global1= np.zeros(ni.shape,dtype = np.float32)

    #heat_info = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],4),dtype = np.float32)
    heat_info = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],3),dtype = np.float32)
    a = []
    u = []
    window_info = []
    laser_flag = 0
    
    #for i_sample in range(deposit_pairs.shape[0]):
    for i_sample in range(10):
        print(i_sample)
        if bjorn:
            i_time_step = num_base + deposit_pairs[i_sample,0] + 1
            vtk_path_00 = osp.join(vtk_dir, f"T{(i_time_step):07}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"T{(i_time_step+1):07d}.vtu")
        else:
            i_time_step = deposit_pairs[i_sample,0]
            vtk_path_00 = osp.join(vtk_dir, f"u_{(i_time_step):05d}_active_{0:05d}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"u_{(i_time_step+1):05d}_active_{0:05d}.vtu")
        if os.path.isfile(vtk_path_00) and os.path.isfile(vtk_path_01):
            print(i_time_step)
            mesh0 = AMMesh(with_base=True,path=vtk_path_00,base_ratio=ratio)
            mesh1 = AMMesh(with_base=True,path=vtk_path_01,base_ratio=ratio)           
            voxel0 = mesh0.to("Voxel")
            voxel1 = mesh1.to("Voxel")          
            voxel0.shift(x_shift,y_shift,z_shift)
            voxel1.shift(x_shift,y_shift,z_shift)                                    
            sol_global0[voxel0.int_coords[:,[0]],voxel0.int_coords[:,[1]],voxel0.int_coords[:,[2]]] = voxel0.voxel_values
            sol_global1[voxel1.int_coords[:,[0]],voxel1.int_coords[:,[1]],voxel1.int_coords[:,[2]]] = voxel1.voxel_values            
            
            print(voxel0.center_coords.shape[0])
            print(voxel1.center_coords.shape[0])
            i_deposit = voxel1.center_coords.shape[0]-num_base                       
            ### update the heat source
            laser_center = voxel1.center_coords[-1]
            heat_info[voxel0.int_coords[:,0],voxel0.int_coords[:,1],voxel0.int_coords[:,2],:3] = laser_center-voxel0.center_coords
            #heat_info[inds_00[:,0],inds_00[:,1],inds_00[:,2],3] = laser_flag

            active_cell_tab[deposit_sequence[:i_deposit+1]] = True 
            rho_global[voxel_inds[active_cell_tab,0],voxel_inds[active_cell_tab,1],\
                       voxel_inds[active_cell_tab,2],:] = 1
 
            sol_global0 *= rho_global[:,:,:,0]
            sol_global1 *= rho_global[:,:,:,0]
            
            for i_ele in range(max(0,i_deposit-num_cut), i_deposit):
                v_ind = voxel_inds[deposit_sequence[i_ele]]
                
                print(i_deposit, i_ele)
                print(f'global v sol00: {sol_global0[v_ind[0],v_ind[1],v_ind[2]]}')
                print(f'global v sol01 {sol_global1[v_ind[0],v_ind[1],v_ind[2]]}')                
                print(f'global v heatinfo00: {heat_info[v_ind[0],v_ind[1],v_ind[2]]}')

                window = (ni<=v_ind[0]+w_radius) & (ni >= v_ind[0]-w_radius) & \
                    (mi <= v_ind[1]+w_radius) &(mi >= v_ind[1]-w_radius) & \
                    (li <= v_ind[2]+w_radius) & (li >= v_ind[2]-w_radius)
                if ni[window].shape[0] == w_size*w_size*w_size:
                    inp_temp = sol_global0[window].reshape((w_size,w_size,w_size,1))                    
                    inp_heat_info = heat_info[window].reshape((w_size,w_size,w_size,-1))
                    inp_bif = bif_global[window].reshape((w_size,w_size,w_size,-1))
                    inp_rho = rho_global[window].reshape((w_size,w_size,w_size,1))         

                    inp = np.concatenate((inp_temp,inp_heat_info,inp_bif,inp_rho), axis=-1)
                    outp_temp = sol_global1[window].reshape((w_size,w_size,w_size,1))

                    a.append([inp])
                    u.append([outp_temp])
                    window_info.append(np.array([[i_deposit,i_ele]]))
    
    a = np.concatenate(a, axis=0)
    print(a.shape)
    u = np.concatenate(u, axis=0)
    print(u.shape)
    window_info = np.concatenate(window_info, axis=0)
    print(window_info.shape)
    ### save 
    data = {
    "a":a,
    "u":u,
    "window_info": window_info,
    }
    return data
    
                
    