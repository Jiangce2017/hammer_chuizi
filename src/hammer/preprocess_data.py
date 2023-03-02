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

def preprocess_data(w_size,model_name,resolution,num_cut,bjorn=False):
    #w_size = 5
    w_radius = (w_size-1)//2

    # load voxels # load ele_sequence

    #model_name = "cone_with_base"
    problem_name = "small_10_base_20"
    femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
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
    #ni,mi,li = np.indices((max_inds[0]+1,max_inds[1]+1,max_inds[2]+1+w_size))
    
    world_res = resolution*2
    ni,mi,li = np.indices((world_res,world_res,world_res+w_size))
    
    ### move the model voxel into the center of world voxel
    x_shift = int((world_res - max_inds[0])//2)
    y_shift = int((world_res - max_inds[1])//2)
    z_shift = int((world_res - max_inds[2])//2)

    ## boundary impact factor (BIF)
    geo_reader = GeoReader(dx,nx,Add_base)
    cad_file = osp.join(geo_dir,model_name+'.stl')
    geo_reader.load_file(cad_file)
    geo_reader.voxelize()
    bif = geo_reader.calculate_bif()
    bif_global = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],2),dtype = np.float32)
    bif_global[voxel_inds[:,0]+x_shift,voxel_inds[:,1]+y_shift,voxel_inds[:,2]+z_shift,:] = bif

    

    ## rho
    rho_global = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],1),dtype = np.float32)
    rho_global[voxel_inds[:,0]+x_shift,voxel_inds[:,1]+y_shift,voxel_inds[:,2]+z_shift,:] = 1


    ## from indices to coordinates

    sol_global_00 = np.zeros(ni.shape,dtype = np.float32)
    sol_global_01= np.zeros(ni.shape,dtype = np.float32)

    #heat_info = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],4),dtype = np.float32)
    heat_info = np.zeros((ni.shape[0],ni.shape[1],ni.shape[2],3),dtype = np.float32)
    a = []
    u = []
    laser_flag = 0
    print(num_base + deposit_pairs + 1)
    for i_sample in range(deposit_pairs.shape[0]):
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
            inds_00[:,0]+=x_shift,
            inds_00[:,1]+=y_shift,
            inds_00[:,2]+=z_shift
            if bjorn:
                sol_00_center = np.expand_dims(mesh_00.cell_data['T'][0],axis=1)
            else:
                sol_00 = mesh_00.point_data['sol']
                sol_00_center = np.mean(sol_00[cells_00],axis=1)
                
            
            sol_global_00[inds_00[:,[0]],inds_00[:,[1]],inds_00[:,[2]]] = sol_00_center ###only works when the model is placed on a base?
            #print(f"global sol00 max: {np.max(sol_global_00)}")
            i_deposit = deposit_pairs[i_sample,1]
            
            
            cells_01 = mesh_01.cells_dict['hexahedron']
            points_01 = mesh_01.points
            if bjorn:
                points_01 /= 1e3
            points_01[points_01[:,2]<0,:] = points_01[points_01[:,2]<0,:]*ratio
            centeroids_01 = np.mean(points_01[cells_01],axis=1)
            centeroids_01_min = np.min(centeroids_01,axis=0,keepdims=True)
            ### from local to global indices
            inds_01 = np.round((centeroids_01-centeroids_01_min)/dx).astype(int)
            inds_01[:,0]+=x_shift,
            inds_01[:,1]+=y_shift,
            inds_01[:,2]+=z_shift
            if bjorn:
                sol_01_center = np.expand_dims(mesh_01.cell_data['T'][0],axis=1)
            else:
                sol_01 = mesh_01.point_data['sol']
                sol_01_center = np.mean(sol_01[cells_01],axis=1)
            
            sol_global_01[inds_01[:,[0]],inds_01[:,[1]],inds_01[:,[2]]] = sol_01_center ###only works when the model is placed on a base?
            #print(f"global sol01 max: {np.max(sol_global_01)}")
            max_idx = np.argmax(sol_global_01)
            #xx_idx*world_res*world_res + yy_idx*world_res+ zz_idx == max_idx
            zz_idx = max_idx % world_res
            yy_idx = ((max_idx-zz_idx) % (world_res*world_res))//world_res
            xx_idx = (max_idx-zz_idx-yy_idx*world_res)//(world_res*world_res)
            #print(np.argmax(sol_global_01))
            #print(xx_idx, yy_idx, zz_idx)
            
            ### update the heat source
            lag = 3
            laser_center = np.array([toolpath[i_deposit,1], toolpath[i_deposit,2], toolpath[i_deposit,3]])
            heat_info[inds_00[:,0],inds_00[:,1],inds_00[:,2],:3] = laser_center-centeroids_00        
            #heat_info[inds_00[:,0],inds_00[:,1],inds_00[:,2],3] = laser_flag
            i_deposit -= lag
            i_load = 0
            for i_ele in range(max(0,i_deposit-num_cut), max(0,i_deposit)):
                p_coord = laser_center
                v_ind = np.round((p_coord-centeroids_01_min)/dx).astype(int)[0]
                v_ind[0]+=x_shift
                v_ind[1]+=y_shift
                v_ind[2]+=z_shift
                #print(v_ind[0], v_ind[1], v_ind[2])
                #v_ind = np.array([xx_idx,yy_idx,zz_idx])

                #print(f'global v sol00: {sol_global_00[v_ind[0],v_ind[1],v_ind[2]]}')
                #print(f'global v sol01 {sol_global_01[v_ind[0],v_ind[1],v_ind[2]]}')

                window = (ni<=v_ind[0]+w_radius) & (ni >= v_ind[0]-w_radius) & \
                    (mi <= v_ind[1]+w_radius) &(mi >= v_ind[1]-w_radius) & \
                    (li <= v_ind[2]+1) & (li >= v_ind[2]-(2*w_radius-1))
                if ni[window].shape[0] == w_size*w_size*w_size:
                    inp_temp = sol_global_00[window].reshape((w_size,w_size,w_size,1))
                    #print(i_time_frame, i_ele, np.max(inp_temp))
                    inp_heat_info = heat_info[window].reshape((w_size,w_size,w_size,-1))
                    inp_bif = bif_global[window].reshape((w_size,w_size,w_size,-1))
                    inp_rho = rho_global[window].reshape((w_size,w_size,w_size,1))         

                    inp = np.concatenate((inp_temp,inp_heat_info,inp_bif,inp_rho), axis=-1)
                    outp_temp = sol_global_01[window].reshape((w_size,w_size,w_size,1))
                    #print(i_time_frame, i_ele, np.max(outp_temp))
                    a.append([inp])
                    u.append([outp_temp])
                    i_load += 1
            print(i_load)
    
    a = np.concatenate(a, axis=0)
    print(a.shape)
    u = np.concatenate(u, axis=0)
    print(u.shape)
    ### save 
    data = {
    "a":a,
    "u":u,
    }
    pickle.dump(data, open( osp.join(ml_data_dir, model_name+'_cut'+str(num_cut)+".pk"), "wb" ))
                
    