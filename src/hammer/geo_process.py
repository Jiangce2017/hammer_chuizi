### load geometry models into hammer
import trimesh
import numpy as np
import pickle
from hammer.utilities.plotting import plot_surf_mesh, plot_toolpath, plot_element_adding

class GeoReader(object):
    def __init__(self,dx:float, nx:int, Add_base_flag=True):
        super(GeoReader, self).__init__() 
        self._dx = dx
        self._nx = nx
        self._Add_base_flag = Add_base_flag
        
        self._geo_file_path = None
        self._geo_mesh = None
        self._ori_vertices = None
        
        self._voxels = None
        self._voxel_trimesh = None
        self._centeroids = None
        self._voxel_inds = None
        self._max_inds = None
        self._base_thickness = -1
        self._base_ind = -1
        self._hex_mesh_points = None
        
        
        self._hex_mesh_cells = None  
        self._deposit_sequence = None
        self._toolpath = None
        self._box_seq_inds_hash = None
        self._box_seq_inds = None
        self._sampled_deposits  = None
        self._deposit_pairs = None
        self._dt = None
        
        # self._load_file()
        # self._normalize()
        # if Add_base_flag:
            # self._add_base()
        
        
        
    def _load_file(self):
        self._geo_mesh = trimesh.load(self._geo_file_path)
        self._ori_vertices = self._geo_mesh.vertices
        
    def load_file(self, geo_file_path):
        self._geo_file_path = geo_file_path
        # self._dx = dx
        # self._nx = nx
        # self._Add_base_flag = Add_base_flag
        
        self._geo_mesh = None
        self._ori_vertices = None
        
        
        self._voxels = None
        self._voxel_trimesh = None
        self._centeroids = None
        self._voxel_inds = None
        self._max_inds = None
        self._base_thickness = -1
        self._base_ind = -1
        self._hex_mesh_points = None
        
        
        self._hex_mesh_cells = None      
        self._deposit_sequence = None
        self._toolpath = None
        self._box_seq_inds_hash = None
        self._box_seq_inds = None
        self._sampled_deposits  = None
        self._deposit_pairs = None
        self._dt = None
        
        self._load_file()
        self._normalize()
        if self._Add_base_flag:
            self._add_base()
        self._generate_timesteps()
        
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
        #xy_len = extend_ratio*self._dx*self._nx
        self._base_thickness = self._dx*self._nx*0.2
        base = trimesh.creation.box((xy_len[0], xy_len[1], self._base_thickness)) ### we can have an identical base for creating dataset
        #base = trimesh.creation.box((xy_len, xy_len, self._base_thickness))
        
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
        self._max_inds = np.max(self._voxel_inds,axis=0)
        self._hex_mesh_points = self._voxel_trimesh.vertices
        
    def get_voxel_mesh(self):
        return self._voxel_trimesh
        
    def get_centeroids(self):
        return self._centeroids
        
    def get_voxel_indx(self):
        return self._voxel_inds
        
    def get_hex_mesh_points(self):
        return self._hex_mesh_points
        
    def get_hex_mesh_cells(self):
        return self._hex_mesh_cells
            
    def get_base_ind(self):
        return self._base_ind
        
    def get_toolpath(self):
        return self._toolpath
        
    def get_sampled_deposits(self):
        return self._sampled_deposits
        
    def get_timesteps(self):
        return self._dt
        
    def _generate_bounding_box_toolpath(self,kwarg:str):
        if kwarg== 'zigzag':
            ### generate the toolpath of the box
            n_x_cubes = self._max_inds[0]+1
            n_xy_cubes = n_x_cubes*(self._max_inds[1]+1)
            n_cubes = n_xy_cubes*(self._max_inds[2]+1)
            
            self._box_seq_inds = np.zeros((n_cubes, 3),dtype=np.int32)
            first_layer_toolpath = np.zeros((n_xy_cubes,3),dtype=np.int32)
            row_b = 0
            for i_y in range(self._max_inds[1]+1):
                if i_y % 2 == 0:
                    first_layer_toolpath[row_b:row_b+n_x_cubes,0] = np.arange(n_x_cubes)
                else:
                    first_layer_toolpath[row_b:row_b+n_x_cubes,0] = np.arange(self._max_inds[0],-1,-1)
                first_layer_toolpath[row_b:row_b+n_x_cubes,1] = i_y
                row_b += n_x_cubes
            self._box_seq_inds[:n_xy_cubes,:] = first_layer_toolpath
            
            second_layer_toolpath = np.flip(first_layer_toolpath,axis=0)
            second_layer_toolpath[:,2] = 1
            self._box_seq_inds[n_xy_cubes:n_xy_cubes*2,:] = second_layer_toolpath
            
            row_b = n_xy_cubes*2
            for i_z in range(2, self._max_inds[2]+1):
                if i_z % 2 == 0:
                    self._box_seq_inds[row_b:row_b+n_xy_cubes,:2] = first_layer_toolpath[:,:2]
                else:
                    self._box_seq_inds[row_b:row_b+n_xy_cubes,:2] = second_layer_toolpath[:,:2]
                self._box_seq_inds[row_b:row_b+n_xy_cubes,2] = i_z
                row_b += n_xy_cubes
        
        #### hash the self._box_seq_inds
        self._box_seq_inds_hash = self._grid_hash(self._box_seq_inds, self._max_inds)
          
    
    def generate_part_toolpath(self,kwarg:str):
        self._generate_bounding_box_toolpath(kwarg)
        voxel_inds_hash = self._grid_hash(self._voxel_inds, self._max_inds)
        model_mask = np.isin(self._box_seq_inds_hash,voxel_inds_hash)
        model_seq_inds = self._box_seq_inds[model_mask]       
        
        ### base inds
        if self._Add_base_flag:        
            self._base_ind = np.ceil((self._base_thickness/self._dx)).astype(int)
        part_mask = model_seq_inds[:,2] > self._base_ind
        toolpath_inds = model_seq_inds[part_mask]
        toolpath_inds_hash = self._grid_hash(toolpath_inds, self._max_inds)
        ### from inds to index in array        
        voxel_lookup_table = self._create_lookup_table(voxel_inds_hash)        
        self._deposit_sequence = voxel_lookup_table[toolpath_inds_hash,1]        
        self._toolpath = np.zeros((self._deposit_sequence.shape[0],5),dtype=np.float32) ### n x 
        self._toolpath[:,1:4] = self._centeroids[self._deposit_sequence]
        
    def sample_deposits(self, num_sample):
        num_d = self._toolpath.shape[0]
        if num_d-1 < num_sample:
            sampled_deposits  = np.arange(1,num_d)
        else:
            arr = np.arange(1,num_d)
            np.random.shuffle(arr)
            arr_selected = arr[:num_sample]
            sampled_deposits  = np.sort(arr_selected)
        deposit_pair = np.zeros((sampled_deposits.shape[0],2),dtype=np.int32)
        deposit_pair[:,1] = sampled_deposits
        deposit_pair[:,0] = sampled_deposits-1
        direction = self._toolpath[deposit_pair[:,1], 1:4] - self._toolpath[deposit_pair[:,0] , 1:4]
        d = np.linalg.norm(direction,axis=1)
        n_step = np.round(d/self._dx)
        self._deposit_pairs = deposit_pair[n_step == 1]
        self._sampled_deposits = np.sort(np.unique(self._deposit_pairs.reshape((-1,1))))
        
    def _generate_timesteps(self):
        self._dt = self._dx/5e-3    
        
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
        print("{} elements".format(hexahedron.shape[0]))
        
    def save_hex_mesh(self, file_path):
        fem_file = {
            "vertices": self._hex_mesh_points,
            "hexahedra": self._hex_mesh_cells,
            "voxel_inds": self._voxel_inds,
            "toolpath": self._toolpath,
            "deposit_sequence":self._deposit_sequence,
            "sampled_deposits":self._sampled_deposits,
            "deposit_pairs":self._deposit_pairs,
            "dt":self._dt,
            "dx":self._dx,
            "nx":self._nx,
            "Add_base":self._Add_base_flag,
        }
        pickle.dump(fem_file,open(file_path, "wb" ))
    
    def calculate_bif(self):
        ## boundary impact factor (BIF)
        bottom_coord = np.min(self._geo_mesh.vertices[:,2])
        centeroids_bif = np.zeros((self._centeroids.shape[0], 2),dtype = np.float32)
        
        centeroids_bif[:,0] = self._centeroids[:,2] - bottom_coord
        
        proximity_query = trimesh.proximity.ProximityQuery(self._geo_mesh)
        centeroids_bif[:,1] = proximity_query.signed_distance(self._centeroids)
        
        return centeroids_bif             
        
    def plot_geo_mesh(self):
        plot_surf_mesh(self._geo_mesh.vertices[self._geo_mesh.faces])  
        
    def plot_part_toolpath(self,toolpath):
        plot_toolpath(toolpath)
        
    def plot_part_element_adding(self, subsample,figure_results_path,movie_path,problem_name):
        plot_element_adding(subsample,self._base_ind,self._deposit_sequence,self._voxel_inds,figure_results_path,movie_path,problem_name)

    def _grid_hash(self,arr,max_inds):
        return arr[:,0]+arr[:,1]*(max_inds[0]+1) +arr[:,2]*(max_inds[0]+1)*(max_inds[1]+1)
        
    def _create_lookup_table(self,arr_hash):
        lookup_table = np.zeros((np.max(arr_hash)+1, 2),dtype=np.int32)
        lookup_table[arr_hash,1] = np.arange(arr_hash.shape[0])
        return lookup_table
        
    def _coord2inds(self, points, d, coord_min):
        return np.round((points-coord_min)/d).astype(int)
        
        
