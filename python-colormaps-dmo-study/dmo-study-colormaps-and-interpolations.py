#!/usr/bin/env python3

#------------------------------------------------------
# Study for different colormaps and interpolations
# for DMO projections
# Loads of colormaps and interpolations
# Turns out with high resolution simulations,
# interpolation is useless.
#------------------------------------------------------

import numpy as np
import scipy.io 
import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Full cmaps list

# from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
cmaps = [   'viridis', 'plasma', 'inferno', 'magma', 'cividis',                                                 # perceptually uniform sequential colormaps
            'Blues', 'Greens', 'Oranges', 'Reds', 'YlGnBu', 'BuPu', 'PuBu', 'GnBu', 'RdPu', 'PuRd', 'OrRd',     # Sequential colormaps
            'binary', 'gist_yarg', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'copper',  # Sequential (2) colormaps
        ]

interpols = [  'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
               'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
               'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
            ]

# reduced lists
cmaps = [   'viridis', 
            'plasma', 
            'inferno', 
            'magma', 
            'cividis',
            'Blues', 
            'Greens', 
            'Oranges', 
            'Reds', 
            #  'YlGnBu',
            #  'BuPu',
            #  'PuBu',
            #  'GnBu',
            #  'RdPu',
            #  'PuRd',
            #  'OrRd',
            #  'binary',
            #  'gist_yarg',
            #  'pink',
            #  'spring',
            #  'summer',
            #  'autumn',
            #  'winter',
            #  'cool',
            #  'Wistia',
            'copper',  
        ]

interpols = [   'none', 
                'nearest', 
                'bilinear', 
                'bicubic', 
                'spline16',
                'spline36', 
                'hanning', 
                'hamming', 
                'hermite', 
                'kaiser', 
                'quadric',
                'catrom', 
                'gaussian', 
                'bessel', 
                'mitchell', 
                'sinc', 
                'lanczos'
            ]


ncmaps = len(cmaps)
ninter = len(interpols)



#================================
# Read data
#================================


mapfile = '../inputfiles/part2map-hires.dat'
if not os.path.exists(mapfile):
    print("I didn't find ", mapfile)
    quit(2)

f = scipy.io.FortranFile(mapfile)

f.read_reals(dtype=np.float64) # t, dx, dy, dz
nx, ny = f.read_ints(dtype=np.int32)

data = f.read_reals(dtype=np.float32)
minval = data[data>0].min()
maxval = data.max()
data[data<minval] = minval/10 # cut off low end
data = data.reshape((nx,ny))

xmin, xmax = f.read_reals(dtype=np.float64)
ymin, ymax = f.read_reals(dtype=np.float64)






#=============================
print("Creating Image")
#=============================


fig = plt.figure(figsize=(ninter*5,ncmaps*5), dpi=200)

counter = 0

for i, cmap in enumerate(cmaps):
    for j, inter in enumerate(interpols):

        counter += 1

        ax=fig.add_subplot(ncmaps, ninter, counter, aspect='equal')

        im = ax.imshow(data,
            interpolation=inter,
            cmap=cmap,
            origin='lower',
            extent=(0,1,0,1),
            norm=LogNorm(),
            vmin=minval/10, vmax=maxval*2
            )

        ax.set_xticks([])
        ax.set_yticks([])


        ax.set_title(inter)
        ax.set_ylabel(cmap)




fig.tight_layout()

figname = 'plot_dmo-study.png'
print("saving figure ", figname)
plt.savefig(figname, form='png', dpi=fig.dpi)
#  figname = 'plot_dmo-study.jpg'
#  print("saving figure ", figname)
#  plt.savefig(figname, progressive=True, quality=70, form='jpg', dpi=fig.dpi)
