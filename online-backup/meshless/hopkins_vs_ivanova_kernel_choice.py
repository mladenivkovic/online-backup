#!/usr/bin/env python3

#===============================================================
# Compute effective surfaces at different particle positions
# for different kernels
#===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size 
import meshless as ms

import h5py


ptype = 'PartType0'             # for which particle type to look for



# border limits for plots
lowlim = 0.40
uplim = 0.60
tol = 1e-5 # tolerance for float comparison
L = 10


kernels = ms.kernels




#========================
def main():
#========================
    
    #-----------------------------
    # Part1 : compute all A
    #-----------------------------

    nx, filenummax, fileskip = ms.get_sample_size()
    #  nx = 10


    nrows = len(kernels)
    ncols = 4

    Aij_Hopkins = [np.zeros((nx,nx,2), dtype=np.float) for k in kernels]
    Aij_Ivanova = [np.zeros((nx,nx,2), dtype=np.float) for k in kernels]
    Aij_Ivanova_v2 = [np.zeros((nx,nx,2), dtype=np.float) for k in kernels]
    Aij_Ivanova_v3 = [np.zeros((nx,nx,2), dtype=np.float) for k in kernels]

    A_list = [Aij_Hopkins, Aij_Ivanova, Aij_Ivanova_v2, Aij_Ivanova_v3]
    A_list_funcs = [ms.Aij_Hopkins, ms.Aij_Ivanova, ms.Aij_Ivanova_analytical_gradients, ms.Aij_Ivanova_approximate_gradients]



    # loop over all files
    ii = 0
    jj = 0
    for i in range(1, filenummax+1, fileskip):
        for j in range(1, filenummax+1, fileskip):


            srcfile = 'snapshot-'+str(i).zfill(3)+'-'+str(j).zfill(3)+'_0000.hdf5'
            print('working for ', srcfile)

            x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)
            H = ms.get_H(h)

            cind = ms.find_central_particle(L, ids)
            pind = ms.find_added_particle(ids)
            nbors = ms.find_neighbours(pind, x, y, H)
            try:
                ind = nbors.index(cind)
            except ValueError:
                print(nbors)
                print(x[nbors])
                print(y[nbors])
                print('central:', cind, x[cind], y[cind])
                print('displaced:', pind, x[pind], y[pind], 2*h[pind])


            for k,kernel in enumerate(kernels):
                print(kernel)
                for a in range(len(A_list)):
                    f = A_list_funcs[a]
                    A = A_list[a]
                    res = f(pind, x, y, H, m, rho, kernel=kernel)
                    A[k][jj,ii] = res[ind]

            
            jj += 1

        ii += 1
        jj = 0
    



    fig = plt.figure(figsize=(4*ncols+0.5, 3.5*nrows+1.5))


    # prepare particles to plot

    inds = np.argsort(ids)
    xp = x[inds][:-1]
    yp = y[inds][:-1]

    mask = np.logical_and(xp>=lowlim-tol, xp<=uplim+tol)
    mask = np.logical_and(mask, yp>=lowlim-tol)
    mask = np.logical_and(mask, yp<=uplim+tol)

    ps = 100
    fc = 'white'
    ec = 'black'
    lw = 2




    counter = 1
    for k, kernel in enumerate(kernels):

        ax1 = fig.add_subplot(nrows, ncols, counter, aspect='equal')
        ax2 = fig.add_subplot(nrows, ncols, counter+1, aspect='equal')
        ax3 = fig.add_subplot(nrows, ncols, counter+2, aspect='equal')
        ax4 = fig.add_subplot(nrows, ncols, counter+3, aspect='equal')
        counter += ncols

        axes = [ax1, ax2, ax3, ax4]


        for i, A in enumerate(A_list):
            
            ax = axes[i]
            Ax = A[k][:,:,0]
            Ay = A[k][:,:,1]
            Anorm = np.sqrt(Ax**2 + Ay**2)
            #  xmin = Ax.min()
            #  xmax = Ax.max()
            #  ymin = Ay.min()
            #  ymax = Ay.max()
            normmin = Anorm.min()
            normmax = Anorm.max()


            # reset lowlim and maxlim so cells are centered around the point they represent
            dx = (uplim - lowlim) / A[k].shape[0]


            uplim2 = uplim
            lowlim2 = lowlim

            cmap = 'YlGnBu_r'

            im = ax.imshow(Anorm, origin='lower', 
                    vmin=normmin, vmax=normmax, cmap=cmap,
                    extent=(lowlim2, uplim2, lowlim2, uplim2))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            fig.colorbar(im, cax=cax)


            # scatter neighbours
            ax.scatter(xp[mask], yp[mask], s=ps, lw=lw,
                    facecolor=fc, edgecolor=ec)

            # scatter central
            ax.scatter(x[cind], y[cind], s=ps, lw=lw,
                    facecolor=ec, edgecolor=ec)

            ax.set_xlim((lowlim2,uplim2))
            ax.set_ylim((lowlim2,uplim2))


        ax1.set_ylabel(kernel, fontsize=12) 

        if k==0:
            ax1.set_title(r'Hopkins |$\mathbf{A}_{ij}$|', fontsize=12, pad=8)

            ax2.set_title(r'Ivanova |$\mathbf{A}_{ij}$|', fontsize=12, pad=8)

            ax3.set_title(r'Ivanova v2 analytic gradients |$\mathbf{A}_{ij}$|', fontsize=12, pad=8)

            ax4.set_title(r'Ivanova v2 approx gradients |$\mathbf{A}_{ij}$|', fontsize=12, pad=8)



    fig.suptitle(r'Effective Area $\mathbf{A}_{ij}$ of a particle w.r.t. the central particle (black) in a uniform distribution for different kernels', fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig('different_kernels.png', dpi=300)

    print('finished.')







    return





if __name__ == '__main__':
    main()

