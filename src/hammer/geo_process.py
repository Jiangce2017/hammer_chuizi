### load geometry models into hammer
import trimesh
import numpy as np
from hammer.utilities.plotting import plot_surf_mesh, plot_toolpath

class GeoReader(object):
    def __init__(self, geo_file_path:str, dx:float, nx:int, add_base=False):
        super(GeoReader, self).__init__()
        self._geo_file_path = geo_file_path
        self._dx = dx
        self._nx = nx
        
        self._geo_mesh = None
        self._ori_vertices = None
        self._voxels = None
        self._voxel_trimesh = None
        self._centeroids = None
        self._voxel_inds = None
        self._hex_mesh_points = None
        self._hex_mesh_cells = None
        self._base_thickness = -1
        
        self._load_file()
        self._normalize()
        if add_base:
            self._add_base()
        
        
        
    def _load_file(self):
        self._geo_mesh = trimesh.load(self._geo_file_path)
        self._ori_vertices = self._geo_mesh.vertices
        
    def load_file(self, geo_file_path, dx, nx, add_base):
        self._geo_file_path = geo_file_path
        self._dx = dx
        self._nx = nx
        
        self._geo_mesh = None
        self._ori_vertices = None
        self._voxels = None
        self._voxel_trimesh = None
        self._centeroids = None
        self._voxel_inds = None
        self._hex_mesh_points = None
        self._hex_mesh_cells = None
        self._base_thickness = -1
        
        self._load_file()
        self._normalize()
        if add_base:
            self._add_base()
        
    def _normalize(self):     
        bounds = self._get_bounds()
        #### move the center at (0,0,0)
        center = np.mean(bounds,axis=0)
        matrix = np.eye(4)
        matrix[:3,3] = -center
        self._geo_mesh.apply_transform(matrix)
        
        #### scale inside the box dx*nx x dx*nx x dx*nx
        len_cube = np.max(bounds[1,:]-bounds[0,:])
        scale = self._dx*self._nx/len_cube
        matrix = np.eye(4)
        matrix[:3,:3] *= scale
        self._geo_mesh.apply_transform(matrix)
        
        #### set the bottom at z=0
        matrix = np.eye(4)
        z_min = np.min(self._geo_mesh.vertices[:,2])
        matrix[2,3] = -z_min
        self._geo_mesh.apply_transform(matrix)
        
    def _get_bounds(self):
        bounds = np.zeros((2,3),dtype=np.float32)
        bounds[0,:] = np.min(self._geo_mesh.vertices,axis=0)
        bounds[1,:] = np.max(self._geo_mesh.vertices,axis=0)
        return bounds
        
    def _add_base(self):
        bounds = self._get_bounds()
        extend_ratio = 1.6
        xy_len = extend_ratio*(bounds[1,:2]-bounds[0,:2])
        self._base_thickness = self._dx*self._nx*0.2
        base = trimesh.creation.box((xy_len[0], xy_len[1], self._base_thickness)) ### we can have an identical base for creating dataset
        
        #### move the base to the center
        center = np.mean(bounds,axis=0)
        
        geo_model_z_min = bounds[0,2]
        t_z = geo_model_z_min + self._dx/8 - self._base_thickness/2 
        matrix = np.eye(4)
        matrix[2,3] = t_z
        matrix[:2,3] = center[:2]
        base.apply_transform(matrix)
        
        self._geo_mesh = trimesh.boolean.union([self._geo_mesh, base])
        
    def voxelize(self):
        self._voxels = self._geo_mesh.voxelized(self._dx, method='subdivide').fill()
        self._voxel_trimesh = self._voxels.as_boxes()
        self._centeroids = self._voxels.points
        self._voxel_inds = self._voxels.points_to_indices(self._centeroids)
        self._hex_mesh_points = self._voxel_trimesh.vertices
        
    def get_voxel_mesh(self):
        return self._voxel_trimesh
        
    def get_centeroids(self):
        return self._centeroids
        
    def get_voxel_indx(self):
        return self._voxel_inds
        
    def get_hex_mesh_points(self):
        return self._hex_mesh_points
        
    def get_hex_mesh_points(self):
        if self._hex_mesh_cells==None:
            raise ValueError('Use generate_hexahedron_cells to generate cells first')
        else:
            return self._hex_mesh_cells
        
    def generate_bounding_box_toolpath(self,max_inds,kwarg:str):
        if kwarg== 'zigzag':
            ### generate the toolpath of the box
            #max_inds = np.max(self._voxel_inds,axis=0)
            n_x_cubes = max_inds[0]+1
            n_xy_cubes = n_x_cubes*(max_inds[1]+1)
            n_cubes = n_xy_cubes*(max_inds[2]+1)
            
            box_seq_inds = np.zeros((n_cubes, 3),dtype=np.int32)
            row_b = 0
            for i_y in range(max_inds[1]+1):
                if i_y % 2 == 0:
                    box_seq_inds[row_b:row_b+n_x_cubes,0] = np.arange(n_x_cubes)
                else:
                    box_seq_inds[row_b:row_b+n_x_cubes,0] = np.arange(max_inds[0],-1,-1)
                box_seq_inds[row_b:row_b+n_x_cubes,1] = i_y
                row_b += n_x_cubes
            first_layer_toolpath = box_seq_inds[:n_xy_cubes,:]
            second_layer_toolpath = np.flip(first_layer_toolpath,axis=0)
            second_layer_toolpath[:,2] += 1
            box_seq_inds[n_xy_cubes:n_xy_cubes*2,:] = second_layer_toolpath
            
            row_b = n_xy_cubes*2
            for i_z in range(2, max_inds[2]+1):
                if i_z % 2 == 0:
                    box_seq_inds[row_b:row_b+n_xy_cubes,:2] = first_layer_toolpath[:,:2]
                else:
                    box_seq_inds[row_b:row_b+n_xy_cubes,:2] = second_layer_toolpath[:,:2]
                box_seq_inds[row_b:row_b+n_xy_cubes,2] = i_z
                row_b += n_xy_cubes
        
        
        #### hash the box_seq_inds
        box_seq_inds_hash = self._grid_hash(box_seq_inds, max_inds)
        
        return box_seq_inds,box_seq_inds_hash
     
    
    def generate_part_toolpath(self,box_seq_inds, box_seq_inds_hash,max_inds):
        voxel_inds_hash = self._grid_hash(self._voxel_inds, max_inds)
        model_mask = np.isin(box_seq_inds_hash,voxel_inds_hash)
        model_seq_inds = box_seq_inds[model_mask]
        
        model_seq_inds_hash = self._grid_hash(model_seq_inds,max_inds)
        
        ### base inds 
        base_ind = np.ceil((self._base_thickness/self._dx)).astype(int)
        part_mask = model_seq_inds[:,2] > base_ind
        toolpath_inds = model_seq_inds[part_mask]
        toolpath_inds_hash = self._grid_hash(toolpath_inds, max_inds)
        ### from inds to index in array        
        voxel_lookup_table = self._create_lookup_table(voxel_inds_hash)
        
        deposit_sequence = voxel_lookup_table[toolpath_inds_hash,1]
        
        ele_sequence = np.zeros((model_seq_inds_hash.shape[0],2),np.int32)
        ele_sequence[:,0] = voxel_lookup_table[model_seq_inds_hash,1]
        ele_sequence[part_mask][1] = 1
        
        toolpath = np.zeros((deposit_sequence.shape[0],5),dtype=np.float32)
        toolpath[:,1:4] = self._centeroids[deposit_sequence]
        
        return toolpath, deposit_sequence, ele_sequence
        
    def generate_hexahedron_cells(self):
        ### convert coordicates to inds
        point_coord_min = np.min(self._hex_mesh_points,axis=0,keepdims=True)
        points_inds = self._coord2inds(self._hex_mesh_points, self._dx,point_coord_min)
        points_max_inds = np.max(points_inds,axis=0)
        points_inds_hash = self._grid_hash(points_inds, points_max_inds)
        points_lookup_table = self._create_lookup_table(points_inds_hash)
        delta = self._dx/2
        moves = delta*np.array([[-1,1,-1],
                                [-1,-1,-1],
                                [1,-1,-1],
                                [1,1,-1],                            
                                [-1,1,1],
                                [-1,-1,1],
                                [1,-1,1],
                                [1,1,1],],np.float16)
        hexahedron = np.zeros((self._centeroids.shape[0],8),dtype=np.int32)
        for i_m in range(8):
            m_p = self._centeroids - moves[[i_m]]
            ### convert coordicates to inds
            m_p_inds = self._coord2inds(m_p, self._dx, point_coord_min)
            m_p_inds_hash = self._grid_hash(m_p_inds, points_max_inds)
            hexahedron[:,i_m] = points_lookup_table[m_p_inds_hash,1]
        self._hex_mesh_cells = hexahedron
    
    def calculate_bif(self):
        pass
        
    def plot_geo_mesh(self):
        plot_surf_mesh(self._geo_mesh.vertices[self._geo_mesh.faces])  
        
    def plot_part_toolpath(self,toolpath):
        plot_toolpath(toolpath)

    def _grid_hash(self,arr,max_inds):
        return arr[:,0]+arr[:,1]*(max_inds[0]+1) +arr[:,2]*(max_inds[0]+1)*(max_inds[1]+1)
        
    def _create_lookup_table(self,arr_hash):
        lookup_table = np.zeros((np.max(arr_hash)+1, 2),dtype=np.int32)
        lookup_table[arr_hash,1] = np.arange(arr_hash.shape[0])
        return lookup_table
        
    def _coord2inds(self, points, d, coord_min):
        return np.round((points-coord_min)/d).astype(int)
        
        
