#!/usr/bin/env python3

#------------------------------------------------------
# Study for different colormaps and interpolations
# for DMO projections
# Loads of colormaps and interpolations
#------------------------------------------------------

import numpy as np
import scipy.io 
import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Full cmaps list

# from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

cmaps = [   'viridis',
            'plasma',
            'inferno',
            'magma',
            'cividis',
            'Greys',
            'Purples',
            'Blues',
            'Greens',
            'Oranges',
            'Reds',
            'YlOrBr',
            'YlOrRd',
            'OrRd',
            'PuRd',
            'RdPu',
            'BuPu',
            'GnBu',
            'PuBu',
            'YlGnBu',
            'PuBuGn',
            'BuGn',
            'YlGn',
            'binary',
            'gist_yarg',
            'gist_gray',
            'gray',
            'bone',
            'pink',
            'spring',
            'summer',
            'autumn',
            'winter',
            'cool',
            'Wistia',
            'hot',
            'afmhot',
            'gist_heat',
            'copper',
            'viridis_r',
            'plasma_r',
            'inferno_r',
            'magma_r',
            'cividis_r',
            'Greys_r',
            'Purples_r',
            'Blues_r',
            'Greens_r',
            'Oranges_r',
            'Reds_r',
            'YlOrBr_r',
            'YlOrRd_r',
            'OrRd_r',
            'PuRd_r',
            'RdPu_r',
            'BuPu_r',
            'GnBu_r',
            'PuBu_r',
            'YlGnBu_r',
            'PuBuGn_r',
            'BuGn_r',
            'YlGn_r',
            'binary_r',
            'gist_yarg_r',
            'gist_gray_r',
            'gray_r',
            'bone_r',
            'pink_r',
            'spring_r',
            'summer_r',
            'autumn_r',
            'winter_r',
            'cool_r',
            'Wistia_r',
            'hot_r',
            'afmhot_r',
            'gist_heat_r',
            'copper_r',
        ]

interpols = [  'nearest' ]



ncmaps = len(cmaps)

#  nrows = int(np.sqrt(ncmaps)+0.5)
#  ncols = nrows
#  while nrows*ncols < ncmaps:
#      ncols += 1

nrows = 8
ncols = 10



#================================
# Read data
#================================


mapfile = 'part2map-hires.dat'
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


fig = plt.figure(figsize=(ncols*5,nrows*5), dpi=200)

counter = 0

for i, cmap in enumerate(cmaps):
    for j, inter in enumerate(interpols):

        counter += 1

        ax=fig.add_subplot(nrows, ncols, counter, aspect='equal')

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


        ax.set_title(cmap)




fig.tight_layout()

figname = 'plot_dmo-study-colormaps.png'
print("saving figure ", figname)
plt.savefig(figname, form='png', dpi=fig.dpi)
#  figname = 'plot_dmo-study.jpg'
#  print("saving figure ", figname)
#  plt.savefig(figname, progressive=True, quality=70, form='jpg', dpi=fig.dpi)
