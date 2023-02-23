import meshio
import os.path as osp
import os
from pathlib import Path
import trimesh
import numpy as np
import pickle
from numpy import linalg as LA

from hammer import GeoReader

data_dir = os.path.join(Path.home(), 'data','hammer') 

def preprocess_data(w_size,model_name,bjorn=False):
    #w_size = 5
    w_radius = (w_size-1)//2

    # load voxels # load ele_sequence

    #model_name = "cone_with_base"
    problem_name = "small_10_base_20"
    femfile_dir = osp.join(data_dir,"meshes",problem_name)
    geo_dir = osp.join(data_dir, "geo_models","small_10")
    if bjorn:
        vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    else:    
        vtk_dir = osp.join(data_dir,"vtk",problem_name, model_name)
    
    ml_data_dir = osp.join(data_dir,"ml_data",model_name)
    os.makedirs(ml_data_dir, exist_ok=True)

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

    max_inds = np.max(voxel_inds,axis=0)
    max_coord = np.max(max_inds)
    ni,mi,li = np.indices((max_inds[0]+1,max_inds[1]+1,max_inds[2]+1+w_size))


    ## boundary impact factor (BIF)
    geo_reader = GeoReader(dx,nx,Add_base)
    cad_file = osp.join(geo_dir,model_name+'.stl')
    geo_reader.load_file(cad_file)
    geo_reader.voxelize()
    bif = geo_reader.calculate_bif()
    bif_global = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],2),dtype = np.float32)
    bif_global[voxel_inds[:,0],voxel_inds[:,1],voxel_inds[:,2],:] = bif

    

    ## rho
    rho_global = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],1),dtype = np.float32)
    rho_global[voxel_inds[:,0],voxel_inds[:,1],voxel_inds[:,2],:] = 1


    ## from indices to coordinates

    sol_global_00 = np.zeros(ni.shape,dtype = np.float32)
    sol_global_01= np.zeros(ni.shape,dtype = np.float32)

    #heat_info = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],4),dtype = np.float32)
    heat_info = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],3),dtype = np.float32)
    a = []
    u = []
    laser_flag = 0
    for i_deposit in range(deposit_pairs.shape[0]):
        print(i_deposit)
        if bjorn:
            i_time_step = num_base + deposit_pairs[i_deposit,0] + 2
            vtk_path_00 = osp.join(vtk_dir, f"T{(i_time_step):07}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"T{(i_time_step+1):07d}.vtu")
        else:
            i_time_step = deposit_pairs[i_deposit,0]
            vtk_path_00 = osp.join(vtk_dir, f"u_{(i_time_step):05d}_active_{0:05d}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"u_{(i_time_step+1):05d}_active_{0:05d}.vtu")
        if os.path.isfile(vtk_path_00) and os.path.isfile(vtk_path_01):
            print(i_time_step)
            mesh_00 = meshio.read(vtk_path_00)
            mesh_01 = meshio.read(vtk_path_01)

            
            cells_00 = mesh_00.cells_dict['hexahedron']
            points_00 = mesh_00.points
            if bjorn:
                points_00 /= 1e3
            
            points_00[points_00[:,2]<0,:] = points_00[points_00[:,2]<0,:]*ratio
            centeroids_00 = np.mean(points_00[cells_00],axis=1)
            centeroids_00_min = np.min(centeroids_00,axis=0,keepdims=True)
            ### from local to global indices
            inds_00 = np.round((centeroids_00-centeroids_00_min)/dx).astype(int)
            if bjorn:
                sol_00_center = np.expand_dims(mesh_00.cell_data['T'][0],axis=1)
            else:
                sol_00 = mesh_00.point_data['sol']
                sol_00_center = np.mean(sol_00[cells_00],axis=1)
            sol_global_00[inds_00[:,[0]],inds_00[:,[1]],inds_00[:,[2]]] = sol_00_center ###only works when the model is placed on a base?
            
            i = deposit_pairs[i_deposit,1]
            ### update the heat source
            laser_center = np.array([toolpath[i,1], toolpath[i,2], toolpath[i,3]])
            heat_info[inds_00[:,0],inds_00[:,1],inds_00[:,2],:3] = laser_center-centeroids_00
            #heat_info[inds_00[:,0],inds_00[:,1],inds_00[:,2],3] = laser_flag
            
            
            cells_01 = mesh_01.cells_dict['hexahedron']
            points_01 = mesh_01.points
            if bjorn:
                points_01 /= 1e3
            points_01[points_01[:,2]<0,:] = points_01[points_01[:,2]<0,:]*ratio
            centeroids_01 = np.mean(points_01[cells_01],axis=1)
            centeroids_01_min = np.min(centeroids_01,axis=0,keepdims=True)
            ### from local to global indices
            inds_01 = np.round((centeroids_01-centeroids_01_min)/dx).astype(int)
            if bjorn:
                sol_01_center = np.expand_dims(mesh_01.cell_data['T'][0],axis=1)
            else:
                sol_01 = mesh_01.point_data['sol']
                sol_01_center = np.mean(sol_01[cells_01],axis=1)
            
            sol_global_01[inds_01[:,[0]],inds_01[:,[1]],inds_01[:,[2]]] = sol_01_center ###only works when the model is placed on a base?

            i_time_frame = i-1
            if i_time_frame > 50:
                for i_ele in range(i_time_frame-50,i_time_frame+1):
                    p_ind = deposit_sequence[i_ele]
                    v_ind = voxel_inds[p_ind]
                    window = (ni<=v_ind[0]+w_radius) & (ni >= v_ind[0]-w_radius) & (mi <= v_ind[1]+w_radius) &\
                        (mi >= v_ind[1]-w_radius) & (li <= v_ind[2]+w_radius) & (li >= v_ind[2]-w_radius)
                    if sol_global_00[window].shape[0] == w_size*w_size*w_size:
                        inp_temp = sol_global_00[window].reshape((w_size,w_size,w_size,1))
                        inp_heat_info = heat_info[window].reshape((w_size,w_size,w_size,-1))
                        inp_bif = bif_global[window].reshape((w_size,w_size,w_size,-1))
                        inp_rho = rho_global[window].reshape((w_size,w_size,w_size,1))         
                        
                        inp = np.concatenate((inp_temp,inp_heat_info,inp_bif,inp_rho), axis=-1)
                        #inp = np.concatenate((inp_temp,inp_heat_info,inp_rho), axis=-1)
                                    
                        outp_temp = sol_global_01[window].reshape((w_size,w_size,w_size,1))
                        
                        a.append([inp])
                        u.append([outp_temp])
    
    a = np.concatenate(a, axis=0)
    print(a.shape)
    u = np.concatenate(u, axis=0)
    print(u.shape)
    ### save 
    data = {
    "a":a,
    "u":u,
    }
    pickle.dump(data, open( osp.join(ml_data_dir, model_name+".pk"), "wb" ))
    
    # laser_flag = 0
    # ### read vtu
    # for i in range(1,toolpath.shape[0]):
        # direction = toolpath[i, 1:4] - toolpath[i - 1 , 1:4]
        # d = np.linalg.norm(direction)
        # n_step = int(np.round(d/path_dx))
        # for i_step in range(n_step-1,n_step):
            # if n_step == 1:
                # vtk_path_00 = osp.join(vtk_dir, f"u_{(i-1):05d}_active_{0:05d}.vtu")       
                # vtk_path_01 = osp.join(vtk_dir, f"u_{i:05d}_active_{0:05d}.vtu")
                # laser_flag = 1
                
            # if n_step > 1 and i_step == 0:
                # vtk_path_00 = osp.join(vtk_dir, f"u_{(i-1):05d}_active_{0:05d}.vtu")
                # vtk_path_01 = osp.join(vtk_dir, f"u_{i:05d}_inactive_{0:05d}.vtu")
                # laser_flag = 0
            # if n_step > 1 and i_step > 0 and i_step < n_step-1:
                # vtk_path_00 = osp.join(vtk_dir, f"u_{i:05d}_inactive_{(i_step-1):05d}.vtu")        
                # vtk_path_01 = osp.join(vtk_dir, f"u_{i:05d}_inactive_{i_step:05d}.vtu")
                # laser_flag = 0
            # if n_step > 1 and i_step == n_step-1:
                # vtk_path_00 = osp.join(vtk_dir, f"u_{i:05d}_inactive_{(i_step-1):05d}.vtu")
                # vtk_path_01 = osp.join(vtk_dir, f"u_{i:05d}_active_{0:05d}.vtu")
                # laser_flag = 1
            # try: 
                # f = open(vtk_path_00)
            # except FileNotFoundError:
                # break
            # try:
                # f = open(vtk_path_01)
            # except FileNotFoundError:
               # break
            # mesh_00 = meshio.read(vtk_path_00)
            # mesh_01 = meshio.read(vtk_path_01)

            # sol_00 = mesh_00.point_data['sol']
            # cells_00 = mesh_00.cells_dict['hexahedron']
            # points_00 = mesh_00.points
            
            # points_00[points_00[:,2]<0,:] = points_00[points_00[:,2]<0,:]*ratio
            # centeroids_00 = np.mean(points_00[cells_00],axis=1)
            # centeroids_00_min = np.min(centeroids_00,axis=0,keepdims=True)
            # ### from local to global indices
            # inds_00 = np.round((centeroids_00-centeroids_00_min)/dx).astype(int)
            # sol_00_center = np.mean(sol_00[cells_00],axis=1)
            # sol_global_00[inds_00[:,[0]],inds_00[:,[1]],inds_00[:,[2]]] = sol_00_center ###only works when the model is placed on a base?
            
            
            # ### update the heat source
            # laser_center = np.array([toolpath[i,1], toolpath[i,2], toolpath[i,3]])
            # heat_info[inds_00[:,0],inds_00[:,1],inds_00[:,2],:3] = centeroids_00 - laser_center
            # heat_info[inds_00[:,0],inds_00[:,1],inds_00[:,2],3] = laser_flag
            
            # sol_01 = mesh_01.point_data['sol']
            # cells_01 = mesh_01.cells_dict['hexahedron']
            # points_01 = mesh_01.points
            # points_01[points_01[:,2]<0,:] = points_01[points_01[:,2]<0,:]*ratio
            # centeroids_01 = np.mean(points_01[cells_01],axis=1)
            # centeroids_01_min = np.min(centeroids_01,axis=0,keepdims=True)
            # ### from local to global indices
            # inds_01 = np.round((centeroids_01-centeroids_01_min)/dx).astype(int)
            # sol_01_center = np.mean(sol_01[cells_01],axis=1)
            # sol_global_01[inds_01[:,[0]],inds_01[:,[1]],inds_01[:,[2]]] = sol_01_center ###only works when the model is placed on a base?

            # i_time_frame = i-1
            # for i_ele in range(i_time_frame,i_time_frame+1):
                # p_ind = deposit_sequence[i_ele]
                # v_ind = voxel_inds[p_ind]
                # window = (ni<=v_ind[0]+w_radius) & (ni >= v_ind[0]-w_radius) & (mi <= v_ind[1]+w_radius) &\
                    # (mi >= v_ind[1]-w_radius) & (li <= v_ind[2]+w_radius) & (li >= v_ind[2]-w_radius)
                # if sol_global_00[window].shape[0] == w_size*w_size*w_size:
                    # inp_temp = sol_global_00[window].reshape((w_size,w_size,w_size,1))
                    # inp_heat_info = heat_info[window].reshape((w_size,w_size,w_size,-1))
                    # inp_bif = bif_global[window].reshape((w_size,w_size,w_size,-1))
                    # inp_rho = rho_global[window].reshape((w_size,w_size,w_size,1))         
                    
                    # inp = np.concatenate((inp_temp,inp_heat_info,inp_bif,inp_rho), axis=-1)
                                
                    # outp_temp = sol_global_01[window].reshape((w_size,w_size,w_size,1))
                    
                    # a.append([inp])
                    # u.append([outp_temp])
                
    