from hammer_data import MeshDataClass, GraphDataClass, VoxelDataClass

import pickle
import meshio
import numpy as onp
import numpy.linalg as LA
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
            
 
class AMMesh(MeshDataClass):
    def __init__(self,with_base=False, bjorn=True,path=None,mesh_data=None,base_ratio=None):
        self.data_class = "Mesh"
        self.points = None ## points coordinates
        self.cells = None
        self.point_values = None
        self.cell_values = None
        self.base_ratio = base_ratio
        
        self.with_base = with_base
        self.bjorn = bjorn
        if path != None:
            self.load_data_source(path)
        elif mesh_data != None:
            self.set_data(mesh_data)

    def load_data_source(self, path:str):
        mesh = meshio.read(path)
        self.points = mesh.points ## points coordinates
        self.cells = mesh.cells_dict['hexahedron']
        self.point_values = None
        if self.bjorn:
            self.cell_values = onp.expand_dims(mesh.cell_data['T'][0], axis=1)
        else:
            sol = mesh.point_data['sol']
            self.point_values = sol
            self.cell_values = onp.mean(sol[self.cells],axis=1)
        if self.bjorn:
            self.points /= 1e3
        if self.base_ratio != None:
            self.points[self.points[:,2]<0,2] = self.points[self.points[:,2]<0,2]*self.base_ratio
    
    def get_data_class(self):
        return self.data_class
    
    
    def to(self,data_class:str)-> object:
        if data_class == "Voxel":
            obj = AMVoxel()
        elif data_class == "Graph":
            obj = AMGraph()
        else:
            raise NotImplementedError  
        data = self.to_data(data_class)
        obj.set_data(data)
        return obj
        
        
    def to_data(self,data_class:str):
        if data_class == "Voxel":
            data = self.mesh2voxel()
            if not self.with_base:
                active_mask = (data["voxel_values"][:, 0] > 0.1) & (data["center_coords"][:, 2] > 0)
                data["center_coords"] = data["center_coords"][active_mask]
                data["int_coords"] = data["int_coords"][active_mask]
                data["voxel_values"] = data["voxel_values"][active_mask]               
        elif data_class == "Graph":
            data = self.mesh2voxel()
            if not self.with_base:
                active_mask = (data["voxel_values"][:, 0] > 0.1) & (data["center_coords"][:, 2] > 0)
                data["center_coords"] = data["center_coords"][active_mask]
                data["int_coords"] = data["int_coords"][active_mask]
                data["voxel_values"] = data["voxel_values"][active_mask]  
            
            voxel = AMVoxel(data)
            data = voxel.voxel2graph()
        else:
            raise NotImplementedError     
        return data
        

        

class AMVoxel(VoxelDataClass):
    # def __init__(self,data_class:str):
        # self.data_class = data_class
        # self.center_coords = None
        # self.int_coords = None
        # self.voxel_values = None
        # self.dx = None

    def load_data_source(self, path:str):
        pass

    def get_data_class(self):
        return self.data_class
        
    def shift(self,x_shift,y_shift,z_shift):
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.z_shift = z_shift
        self.int_coords[:,0]+=x_shift
        self.int_coords[:,1]+=y_shift
        self.int_coords[:,2]+=z_shift
        


        
class AMGraph(GraphDataClass):
    # def __init__(self,data_class:str):
        # self.data_class = data_class
        # self.node_features = None
        # self.edge_index = None

    def load_data_source(self, path:str):
        pass

    def get_data_class(self):
        return self.data_class    
          
    def plot_graph(self):
        points = self.node_features[:,:3]
        edge_index = self.edge_index
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p1s = points[edge_index[0, :]]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], '.r', markersize=3)
        p2s = points[edge_index[1, :]]
        ls = onp.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=0.5, colors='b')
        ax.add_collection(lc)
        plt.show()
        
    def torch_data(self):
        node_features = torch.from_numpy(self.node_features)
        edge_index = torch.from_numpy(self.edge_index)
        return node_features, edge_index