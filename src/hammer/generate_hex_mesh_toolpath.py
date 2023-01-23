### generate hexahedron mesh and toolpath given a CAD model
import trimesh
import numpy as np
import pickle
import os.path as osp
from hammer.utilities.plotting import plot_surf_mesh
    
def order_hex_p_indx(cell_p_indx, cell_p,c_p,pitch):
    ordered_indx = []
    delta = pitch/2
    moves = delta*np.array([[1,-1,-1],
                            [1,1,-1],
                            [-1,1,-1],
                            [-1,-1,-1],
                            [1,-1,1],
                            [1,1,1],
                            [-1,1,1],
                            [-1,-1,1]],np.float16)
    for i_m in range(8):
        m_p = c_p - moves[[i_m]]
        ordered_indx.append(cell_p_indx[np.sum((cell_p - m_p)**2,axis=1)<pitch**2*0.1][0])
    return ordered_indx

def generate_hex_mesh_toolpath(cad_file, m, flag_plot_cad=False,flag_plot_hex_mesh=False,):    
    mesh = trimesh.load(cad_file)
    
    if flag_plot_cad:
        plot_surf_mesh(mesh.vertices[mesh.faces])

    bounds = mesh.bounds
    #center_p = np.mean(bounds,axis=0,keepdims=False)
    #corners = trimesh.bounds.corners(bounds)
    len_cube = np.max(bounds[1,:]-bounds[0,:])
    #radius = 15
    #m = 2*radius+1 # resolusion (m,m,m)
    pitch = len_cube/m # Side length of a single voxel cube

    voxels = mesh.voxelized(pitch, method='subdivide').fill()


    voxel_mesh = voxels.as_boxes()
    centeroids = voxels.points
    voxel_inds = voxels.points_to_indices(centeroids)
    ### plot the centeroids one by one
    max_inds = np.max(voxel_inds,axis=0)

    toolpath = []
    hexahedra = []
    ele_sequence = []
    n_v = voxel_mesh.vertices.shape[0]
    mesh_v_ind_array = np.arange(n_v)

    ### change toolpath bellow zigzag
    for i_z in range(max_inds[2]+1):
        for i_y in range(max_inds[1]+1):
            if i_y % 2 == 0:
                i_x_list = range(max_inds[0]+1)
            else:
                i_x_list = range(max_inds[0],-1,-1)
            for i_x in i_x_list:
                if [i_x,i_y,i_z] in voxel_inds.tolist():
                    # find the hexahedron cell
                    p_ind = np.where((voxel_inds==(i_x,i_y,i_z)).all(axis=1))
                    center_coord = centeroids[p_ind]
                    cell_p_mask = np.sum((center_coord-voxel_mesh.vertices)**2,axis=1)< 3/4*pitch**2*1.1
                    cell_p = voxel_mesh.vertices[cell_p_mask]
                    cell_p_indx = mesh_v_ind_array[cell_p_mask]
                    # order the cell_p_indx
                    cell_p_indx_ordered = order_hex_p_indx(cell_p_indx, cell_p, center_coord,pitch)
                    hexahedra.append(cell_p_indx_ordered)
                    # generate the toolpath
                    toolpath.append([center_coord[0,0],center_coord[0,1],center_coord[0,2],1])
                    ele_sequence.append(p_ind[0][0])


    # save mesh and toopath files
    toolpath = np.array(toolpath,np.float16)
    fem_file = {
        "vertices": voxel_mesh.vertices,
        "hexahedra": hexahedra,
        "centeroids": centeroids,
        "ele_sequence": ele_sequence
    }
    #pickle.dump(fem_file, open( "./"+exp_name+"_mesh.p", "wb" ))
    #np.savetxt("./"+exp_name+"_toolpath.txt", toolpath, delimiter=',')

    ## visualize
    if flag_plot_hex_mesh:
        plot_surf_mesh(voxel_mesh.vertices[voxel_mesh.faces])
        
    return fem_file, toolpath, voxels