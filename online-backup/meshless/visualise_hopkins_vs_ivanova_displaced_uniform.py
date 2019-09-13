#!/usr/bin/env python3


#===============================================================
# Visualize effectife area for a chosen particle at various
# positions within an otherwise uniform grid with respect to
# the particle at (0.5, 0.5).
#===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size, ImageGrid


import meshless as ms
from my_utils import setplotparams_multiple_plots

setplotparams_multiple_plots(wspace=0.2)




#---------------------------
# initialize variables
#---------------------------


# temp during rewriting
ptype = 'PartType0'                 # for which particle type to look for
pcoord = [0.5, 0.5]                 # coordinates of particle to work for

L = 10

# border limits for plots
lowlim = 0.4
uplim = 0.6
tol = 1e-2 # tolerance for float comparison

nrows = 2
ncols = 3





#========================
def main():
#========================
    

    nx, filenummax, fileskip =  ms.get_sample_size()
    

    #-----------------------------
    # Part1 : compute all A
    #-----------------------------

    print("Computing effective surfaces")

    AH = np.zeros((nx, nx, 2), dtype=np.float) # storing computed effective surfaces Hopkins
    AI = np.zeros((nx, nx, 2), dtype=np.float) # storing computed effective surfaces ivanova
    #  AI2 = np.zeros((nx, nx, 2), dtype=np.float) # storing computed effective surfaces ivanova
    #  AI3 = np.zeros((nx, nx, 2), dtype=np.float) # storing computed effective surfaces ivanova

    ii = 0
    jj = 0
    for i in range(1, filenummax+1, fileskip):
        for j in range(1, filenummax+1, fileskip):


            srcfile = 'snapshot-'+str(i).zfill(3)+'-'+str(j).zfill(3)+'_0000.hdf5'
            print('working for ', srcfile)

            x, y, h, rho, m, ids, npart= ms.read_file(srcfile, ptype)
            #  AH[jj,ii] = np.random.uniform()
            #  AI[jj,ii] = np.random.uniform()
            #  jj += 1
            #  pind = 10
            #  continue

            H = ms.get_H(h)


            cind = ms.find_central_particle(L, ids)
            pind = ms.find_added_particle(ids)
            # displaced particle has index -1
            nbors = ms.find_neighbours(pind, x, y, H)
            ind = nbors.index(cind)

            Aij = ms.Aij_Hopkins(pind, x, y, H, m, rho)

            # pyplot.imshow takes [y,x] !
            AH[jj, ii] = Aij[ind]

            Aij = ms.Aij_Ivanova(pind, x, y, H, m, rho)
            AI[jj, ii] = Aij[ind]


            jj += 1
        jj = 0
        ii += 1
    





    #-----------------------------
    # Part2: Plot results
    #-----------------------------

    print("Plotting")


    AHx = AH[:,:,0]
    AHy = AH[:,:,1]
    AHnorm = np.sqrt(AHx**2 + AHy**2)
    Hxmin = AHx.min()
    Hxmax = AHx.max()
    Hymin = AHy.min()
    Hymax = AHy.max()
    Hnormmin = AHnorm.min()
    Hnormmax = AHnorm.max()

    AIx = AI[:,:,0]
    AIy = AI[:,:,1]
    AInorm = np.sqrt(AIx**2 + AIy**2)
    Ixmin = AIx.min()
    Ixmax = AIx.max()
    Iymin = AIy.min()
    Iymax = AIy.max()
    Inormmin = AInorm.min()
    Inormmax = AInorm.max()

    imgdata = [[AHx, AHy, AHnorm], [AIx, AIy, AInorm]]


    minval = min(Hnormmin, Hymin, Hxmin, Inormmin, Ixmin, Iymin)
    maxval = max(Hnormmax, Hymax, Hxmax, Inormmax, Ixmax, Iymax)


    # reset lowlim and maxlim so cells are centered around the point they represent
    dx = (uplim - lowlim) / AH.shape[0]

    uplim_plot = uplim #+ 0.5*dx
    lowlim_plot = lowlim # - 0.5*dx




    fig = plt.figure(figsize=(ncols*5, nrows*5.5))

    inds = np.argsort(ids)

    mask = np.logical_and(x>=lowlim-tol, x<=uplim+tol)
    mask = np.logical_and(mask, y>=lowlim-tol)
    mask = np.logical_and(mask, y<=uplim+tol)
    mask[pind] = False

    ec = 'black'
    lw = 2
    cmap = 'YlGnBu_r'



    axrows = [[] for r in range(nrows)]
    for row in range(nrows):
        axcols = [None for c in range(ncols)]

        axcols = ImageGrid(fig, (nrows, 1, row+1),
                    nrows_ncols=(1, ncols), 
                    axes_pad = 0.,
                    share_all = False,
                    label_mode = 'L',
                    cbar_mode = 'edge',
                        cbar_location = 'right',
                        cbar_size = "7%",
                        cbar_pad = "2%")
        axrows[row] = axcols


        for col, ax in enumerate(axcols):
        
            im = ax.imshow(imgdata[row][col], origin='lower', 
                vmin=minval, vmax=maxval, cmap=cmap,
                extent=(lowlim_plot, uplim_plot, lowlim_plot, uplim_plot),
                #  norm=matplotlib.colors.SymLogNorm(1e-3),
                zorder=1)


            # superpose particles

            # plot neighbours (and the ones you drew anyway)
            ps = 100
            fc = 'white'
            ax.scatter(x[mask], y[mask], s=ps, lw=lw,
                    facecolor=fc, edgecolor=ec, zorder=2)
      
            # plot the chosen one
            ps = 150
            fc = 'black'
            ax.scatter(x[cind], y[cind], s=ps, lw=lw,
                    facecolor=fc, edgecolor=ec, zorder=3)
           


            ax.set_xlim((lowlim_plot,uplim_plot))
            ax.set_ylim((lowlim_plot,uplim_plot))
            if col == ncols - 1:
                ax.set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
            else:
                ax.set_xticks([0.4, 0.45, 0.5, 0.55])
            ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6])



            if col == 0:
                ax.set_title(r'$x$ component of $\mathbf{A}_{ij}$', fontsize=14)
                if row == 0:
                    ax.set_ylabel('Hopkins', fontsize=14)
                if row == 1:
                    ax.set_ylabel('Ivanova', fontsize=14)
            if col == 1:
                ax.set_title(r'$y$ component of $\mathbf{A}_{ij}$', fontsize=14)
            if col == 2:
                ax.set_title(r'$|\mathbf{A}_{ij}|$', fontsize=14)



        # Add colorbar to every row
        axcols.cbar_axes[0].colorbar(im)



    fig.suptitle(r'Effective Area $\mathbf{A}_{ij}$ of a particle w.r.t. the central partilce in a uniform distribution')
    plt.savefig('effective_area_displaced_particle.png')

    print('finished.')

    return





if __name__ == '__main__':
    main()

