from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

# Function given in GRA-4152 repo
def plot_grid(images,N=10,C=10,figsize=(24., 28.), color = "bw", name='posterior'):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(N, C),  
                     axes_pad=0,  # pad between Axes in inch.
                     )
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('./xhat_'+color+"_"+name+'.pdf')
    plt.close()