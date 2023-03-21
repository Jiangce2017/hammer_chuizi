import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import imageio
import numpy as np
import os.path as osp
import trimesh
import pickle


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


########################################################################################################################
# TESTING
import plotly.graph_objects as go
from plotly_voxel_display.VoxelData import VoxelData
from matplotlib import cm
import matplotlib


def plotly_fig(subsample, title):
    # subsample: 3D array

    # Establish the 3D Plot axes (modified for non-cubic windows)
    x_axis = np.shape(subsample)[0]
    y_axis = np.shape(subsample)[1]
    z_axis = np.shape(subsample)[2]
    X, Y, Z = np.mgrid[0:1:(x_axis*1j), 0:1:(y_axis*1j), 0:1:(z_axis*1j)]

    # Determine aspect ratio of plot
    minimum = min(x_axis, y_axis, z_axis)
    x_ratio = x_axis/minimum
    y_ratio = y_axis/minimum
    z_ratio = z_axis/minimum

    # Establish the Maximum Value
    maximum = np.amax(subsample)

    # Write the Figure
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=subsample.flatten(),
        isomin=1,  # This establishes the minimum value that is displayed
        isomax=maximum,
        opacity=0.1,  # Needs to be small to see through all surfaces
        surface_count=100, # Larger values mean more surfaces to define the shape
        # slices_z=dict(show=True, locations=[0.4]), # add slice to better show color variation in top view
        colorscale='Bluered',  # Possible color values: 'haline' #Blackbody,Bluered,Blues,Cividis,Earth,Electric,Greens,
                               # Greys,Hot,Jet,Picnic,Portl and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.
        ))
    fig.update_scenes(aspectmode='manual', aspectratio_x=x_ratio, aspectratio_y=y_ratio, aspectratio_z=z_ratio,
                      xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False)
    fig.update_layout(
        scene={
            # 'zaxis': {'autorange': 'reversed'} # reverse automatically)
            },
        title=title
        )
    fig.show()
    return fig


def mesh_3D_voxels(arr, title):
    Voxels = VoxelData(arr)

    fig = go.Figure(data=go.Mesh3d(
            x=Voxels.vertices[0],
            y=Voxels.vertices[1],
            z=Voxels.vertices[2],
            i=Voxels.triangles[0],
            j=Voxels.triangles[1],
            k=Voxels.triangles[2],
            intensity=arr.flatten(),
            colorscale='Bluered',
            opacity=0.75,
            # isomin=1,  # This establishes the minimum value that is displayed
            # isomax=2000,
            ))
    fig.update_layout(
            # scene={
            #     'zaxis': {'autorange': 'reversed'} # reverse automatically)
            #     },
            scene=dict(aspectmode='data'),
            title=title
            )
    fig.show()
    return fig


def matplot_voxels(arr, color_bar_label, title=None, minimum=None, maximum=None, cmap='bwr',subplot=None, fig=None, fontsize=12):
    if subplot == None:
        # Create a new figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot(*subplot, projection='3d')
    #

    # Plot the voxels
    cmap = plt.get_cmap(cmap)

    if minimum == None:
        minimum = np.amin(arr)
    if maximum == None:
        maximum = np.amax(arr)

    norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])

    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(m, cax=cax, aspect=0.5)
    cbar.ax.tick_params(labelsize=fontsize/2)
    cbar.set_label(color_bar_label, fontsize=fontsize/2)
    plt.ticklabel_format(style="plain")
    ax.voxels(arr, edgecolor="k", facecolors=cmap(norm(arr)), alpha=0.5)
    # ax.invert_zaxis()

    # Display the plot
    if title != None:
        ax.set_title(title, fontsize=fontsize)

    ax.tick_params(axis='x', labelsize=fontsize/2)
    ax.tick_params(axis='y', labelsize=fontsize / 2)
    ax.tick_params(axis='z', labelsize=fontsize / 2)
    return ax, fig

########################################################################################################################
def plot_data(true, prediction, windows, time_steps, sample_indices, file_name, fig, fontsize, worst_error=False):
    count = 0
    columns = 4
    number_samples = len(sample_indices)
    if worst_error:
        num_samples = true.shape[0]
        true_u = true.reshape(num_samples, -1)
        pred_u = prediction.reshape(num_samples, -1)
        mse = np.mean((pred_u - true_u) ** 2, axis=1)

        worst_sample_indices = np.argsort(mse)[-number_samples:]
        sample_indices = worst_sample_indices



    for row, index in enumerate(sample_indices):

        # activation_sample = activation[index]
        # a_sample = a[index]
        u_pred_sample = prediction[index]
        u_true_sample = true[index]
        window_sample = windows[index]
        time_step_sample = time_steps[index]

        # List titles
        if row == 0:  # Only plot the titles for the first row
            title1 = "True"
            title2 = "Predicted"
            title3 = "Difference"
            title4 = "Percent Error"
        else:
            title1 = title2 = title3 = title4 = None
        # Plot the True Sample
        count += 1
        ax1, fig1 = matplot_voxels(u_true_sample, "Temperature",title=title1, subplot=(number_samples, columns, count), fig=fig,
                                   fontsize=fontsize)
        ax1.text(-18, -18, 4,
                 "Sample: " + str(index) + "\nTime step: " + str(time_step_sample) + "\nWindow: " + str(window_sample),
                 verticalalignment='center', fontsize=fontsize)

        # Plot the predicted sample
        count += 1
        ax2, fig2 = matplot_voxels(u_pred_sample, "Temperature",title=title2, subplot=(number_samples, columns, count), fig=fig,
                                   fontsize=fontsize)

        # Plot Difference
        error = u_true_sample - u_pred_sample
        count += 1
        ax3, fig3 = matplot_voxels(error, "Temperature", title=title3, subplot=(number_samples, columns, count), fig=fig, fontsize=fontsize)

        # Plot Percent Error
        # Calculate the percentage errors, handling the case where actual is 0
        u_error = np.where(u_true_sample == 0, 0, np.divide(abs(error), abs(u_true_sample)) * 100)
        count += 1
        ax4, fig4 = matplot_voxels(u_error, "% Error", title=title4, subplot=(number_samples, columns, count), fig=fig,
                                   fontsize=fontsize, cmap='Reds')

    plt.savefig("figures/" + file_name)
