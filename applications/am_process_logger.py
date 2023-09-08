import numpy as onp
from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.basis import get_face_shape_vals_and_grads


class am_process_logger(object):
    def __init__(self,full_mesh, active_mesh,points_map_active,cells_map_full,active_cell_truth_tab,ele_type):
        self.full_mesh = full_mesh
        self.active_mesh = active_mesh
        self.points_map_active = points_map_active
        self.cells_map_full = cells_map_full
        self.active_cell_truth_tab = active_cell_truth_tab
        self.active_cell_truth_tab_old = active_cell_truth_tab
        #self.external_faces = external_faces
        #self.hash_map = hash_map
        #self.inner_faces = inner_faces
        #self.all_faces = all_faces
        #self.cells_face = cells_face
        self.ele_type = ele_type
        
    def initialize_hash_map(self):
        #print(f"Initializing hash map for external faces...")
         # (num_faces, num_face_vertices)
        _, _, _, _, face_inds = get_face_shape_vals_and_grads(self.ele_type)
        #face_inds = onp.array(face_inds)
        self.cells_face = self.full_mesh.cells[:, face_inds] # (num_cells, num_faces, num_face_vertices)
        self.cells_face = onp.sort(self.cells_face)
        self.hash_map = {}
        #self.external_faces = onp.empty((0,2),int)
        cell_inds = onp.arange(len(self.cells_face))
        self.hash_map_for_faces(cell_inds)
        #self.external_faces[:, 0] = self.cells_map_full[self.external_faces[:, 0]]
        
        
    def get_active_mesh(self):
        active_cell_inds = onp.argwhere(self.active_cell_truth_tab).reshape(-1)
        cell_map_active = onp.sort(active_cell_inds)
        active_cells = self.full_mesh.cells[cell_map_active]
        self.cells_map_full = onp.zeros(len(self.full_mesh.cells), dtype=onp.int32)
        self.cells_map_full[cell_map_active] = onp.arange(len(cell_map_active))
        active_points_truth_tab = onp.zeros(len(self.full_mesh.points), dtype=bool)
        active_points_truth_tab[active_cells.reshape(-1)] = True
        self.points_map_active = onp.argwhere(active_points_truth_tab).reshape(-1)
        points_map_full = onp.zeros(len(self.full_mesh.points), dtype=onp.int32)
        points_map_full[self.points_map_active] = onp.arange(len(self.points_map_active))
        active_cells = points_map_full[active_cells]
        active_points = self.full_mesh.points[active_points_truth_tab]
        self.active_mesh = Mesh(active_points, active_cells)
        
    def update_hash_map(self):
        new_born_cell_inds = onp.argwhere(self.active_cell_truth_tab_old != self.active_cell_truth_tab).reshape(-1)
        self.hash_map_for_faces(new_born_cell_inds)
        #self.external_faces[:, 0] = self.cells_map_full[self.external_faces[:, 0]]
        
    def hash_map_for_faces(self, cell_inds):
        """Use a hash table to store inner faces 
        """
        #self.inner_faces = onp.empty((0,2),int)
        #self.all_faces = onp.empty((0,2),int)
        self.inner_faces = []
        self.all_faces = []
        for i, cell_id in enumerate(cell_inds):
            if self.active_cell_truth_tab[cell_id]:
                for face_id in range(len(self.cells_face[cell_id])):
                    key = tuple(self.cells_face[cell_id, face_id].tolist())
                    if key in self.hash_map.keys():
                        #self.inner_faces.append(self.hash_map[key])
                        self.inner_faces.append((cell_id, face_id))
                        #self.inner_faces = onp.append(self.inner_faces, onp.array([[cell_id,face_id]],int),axis=0)
                        del self.hash_map[key]
                    else:
                        self.hash_map[key] = (cell_id, face_id)
                    self.all_faces.append((cell_id, face_id))
                    #self.all_faces = onp.append(self.all_faces, onp.array([[cell_id,face_id]],int),axis=0)
        self.external_faces = onp.array(list((set(self.all_faces) - set(self.inner_faces))))
        #external_faces_new = onp.array(list(set(map(tuple, self.all_faces)) - set(map(tuple, self.inner_faces))))
        #external_faces_new[:, 0] = self.cells_map_full[external_faces_new[:, 0]]
        #self.external_faces = onp.append(self.external_faces, external_faces_new, axis=0)