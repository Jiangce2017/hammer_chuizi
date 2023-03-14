import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import imageio
import numpy as np
import os.path as osp
import trimesh

def plot_surf_mesh(tris):
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure,auto_add_to_figure=False)
    figure.add_axes(axes)
    # Add the mesh to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(tris,color='b',alpha=0.4))
    
    # Auto scale to the mesh size
    scale = tris.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel('X axis')
    axes.set_ylabel('Y axis')
    axes.set_zlabel('Z axis')
    #plt.axis('off')
    # Show the plot to the screen
    plt.show()
    
def plot_element_adding(subsample,base_ind,deposit_sequence,voxel_inds,figure_results_path,movie_path,exp_name):
    max_inds = np.max(voxel_inds,axis=0)
    max_coord = np.max(max_inds)

    x,y,z = np.indices((max_inds[0]+1,max_inds[1]+1,max_inds[2]+1))
    colors = np.empty(x.shape,dtype=object)

    cube_i = np.empty(x.shape,dtype=bool)& False
    
    cube_base = z <= base_ind
    colors[cube_base] = 'blue'
    
    images = []
    for i in range(deposit_sequence.shape[0]):
        p_ind = deposit_sequence[i]
        cube_i = cube_i |( (x==voxel_inds[p_ind,0])&(y==voxel_inds[p_ind,1])&(z==voxel_inds[p_ind,2]))
        if i % subsample == 0:
            colors[cube_i] = 'red'
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.voxels(cube_i|cube_base,facecolors=colors, edgecolor='k')
            ax.set_xlim(0,max_coord)
            ax.set_ylim(0,max_coord)
            ax.set_zlim(0,max_coord)
            #plt.show()
            output_file = osp.join(figure_results_path,str(i)+exp_name+'.png')
            plt.savefig(output_file)
            plt.close(fig)
            images.append(imageio.v2.imread(output_file))
        

    imageio.mimsave(movie_path, images)
    
def plot_toolpath(toolpath):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for i in range(toolpath.shape[0]-1):
        ax.plot(toolpath[i:i+2, 1], 
                toolpath[i:i+2, 2], 
                toolpath[i:i+2, 3], 
                color='red' if toolpath[i+1,4]==1 else 'blue',
                marker = 'o')
    fig.tight_layout()
    plt.show()
    
def plot_toolpath_with_voxels(toolpath, voxels):

    voxel_mesh = voxels.as_boxes()
    tris = voxel_mesh.vertices[voxel_mesh.faces]
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure,auto_add_to_figure=False)
    figure.add_axes(axes)
    # Add the mesh to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(tris,color='b',alpha=0.1))
    
    # Auto scale to the mesh size
    scale = tris.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel('X axis')
    axes.set_ylabel('Y axis')
    axes.set_zlabel('Z axis')
    #plt.axis('off')
    # Show the plot to the screen
    for i in range(toolpath.shape[0]-1):
        axes.plot(toolpath[i:i+2, 1], 
                toolpath[i:i+2, 2], 
                toolpath[i:i+2, 3], 
                color='red' if toolpath[i+1,4]==1 else 'blue',
                marker = 'o')
    
    plt.show()
    
def plot_test_voxel(voxel_interpolation):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = plt.get_cmap("gray")
    ax.voxels(voxel_interpolation, edgecolor="k", facecolors=cmap(voxel_interpolation))
    plt.show()
        