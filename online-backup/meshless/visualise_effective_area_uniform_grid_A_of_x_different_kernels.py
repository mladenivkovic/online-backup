#!/usr/bin/env python3

#===============================================================
# Compute A(x) between two specified particles at various
# positions x
#===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size 
import meshless as ms

import h5py


ptype = 'PartType0'             # for which particle type to look for
iind = None                     # index of particle at (0.4, 0.4)
jind = None                     # index of particle in the center (0.5, 0.5)


srcfile = 'snapshot_0000.hdf5'
L = 10      # nr of particles along one axis
boxSize = 1

# border limits for plots
lowlim = 0.35
uplim = 0.55
nx = 50
tol = 1e-5 # tolerance for float comparison


kernels = ms.kernels
kfacts = ms.kernelfacts

#  kernels = ['cubic_spline', 'quintic_spline',
        #  'gaussian', 'gaussian_compact', 'supergaussian',
        #  'wendland_C2', 'wendland_C4', 'wendland_C6']


#========================
def main():
#========================
    
    #-----------------------------
    # Part1 : compute all A
    #-----------------------------

    print("Computing effective surfaces")

    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)

    # find where particles i (0.4, 0.4) and j (0.5, 0.5) are
    iind = None
    jind = None

    for i in range(x.shape[0]):
        if abs(x[i] - 0.4) < tol and abs(y[i] - 0.4) < tol:
            iind = i
        if abs(x[i] - 0.5) < tol and abs(y[i] - 0.5) < tol:
            jind = i


    if iind is None or jind is None:
        raise ValueError("iind=", iind, "jind=", jind)




    #----------------------
    # Set up figure
    #----------------------

    nrows = len(kernels)
    ncols = 3

    fig = plt.figure(figsize=(12.5,3.5*nrows))

    axrows = []
    i = 1
    for r in range(nrows):
        axcols = []
        for c in range(ncols):
            axcols.append(fig.add_subplot(nrows, ncols, i, aspect='equal'))
            i+=1
        axrows.append(axcols)





    #------------------
    # loop and compute
    #------------------


    for k,kernel in enumerate(kernels):

        print('working for ', kernel)

        A = np.zeros((nx, nx, 2), dtype=np.float) # storing computed effective surfaces
        dx = (uplim - lowlim)/nx


        #---------------------
        # compute A
        #---------------------

        for i in range(nx):
            xx = lowlim + dx * i

            if i%10==0:
                print("i =", i, "/", nx)

            for j in range(nx):
                yy = lowlim + dx * j

                hh = ms.h_of_x(xx, yy, x, y, h, m, rho, kernel=kernel)

                A[j,i] = ms.Integrand_Aij_Ivanova(iind, jind, xx, yy, hh, x, y, h, m, rho, kernel=kernel, fact=kfacts[k]) # not a typo: need A[j,i] for imshow


        #---------------------
        # now plot it
        #---------------------

        Ax = A[:,:,0] 
        Ay = A[:,:,1]
        Anorm = np.sqrt(Ax**2 + Ay**2)
        xmin = Ax.min()
        xmax = Ax.max()
        ymin = Ay.min()
        ymax = Ay.max()
        normmin = Anorm.min()
        normmax = Anorm.max()


        # reset lowlim and maxlim so cells are centered around the point they represent
        dx = (uplim - lowlim) / A.shape[0]


        #  uplim2 = uplim - 0.005
        #  lowlim2 = lowlim + 0.005
        uplim2 = uplim
        lowlim2 = lowlim

        cmap = 'YlGnBu_r'


        axcols = axrows[k]
        ax1 = axcols[0]
        ax2 = axcols[1]
        ax3 = axcols[2]

        im1 = ax1.imshow(Ax, origin='lower', 
                vmin=xmin, vmax=xmax, cmap=cmap,
                extent=(lowlim2, uplim2, lowlim2, uplim2))
        im2 = ax2.imshow(Ay, origin='lower', 
                vmin=ymin, vmax=ymax, cmap=cmap,
                extent=(lowlim2, uplim2, lowlim2, uplim2))
        im3 = ax3.imshow(Anorm, origin='lower', 
                vmin=normmin, vmax=normmax, cmap=cmap,
                extent=(lowlim2, uplim2, lowlim2, uplim2))

        for ax, im in [(ax1, im1), (ax2, im2), (ax3, im3)]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            fig.colorbar(im, cax=cax)

        # superpose particles

        inds = np.argsort(ids)

        mask = np.logical_and(x>=lowlim-tol, x<=uplim+tol)
        mask = np.logical_and(mask, y>=lowlim-tol)
        mask = np.logical_and(mask, y<=uplim+tol)

        ps = 50
        fc = 'grey'
        ec = 'black'
        lw = 2

        # plot neighbours (and the ones you drew anyway)
        ax1.scatter(x[mask], y[mask], s=ps, lw=lw,
                facecolor=fc, edgecolor=ec)
        ax2.scatter(x[mask], y[mask], s=ps, lw=lw, 
                facecolor=fc, edgecolor=ec)
        ax3.scatter(x[mask], y[mask], s=ps, lw=lw,
                facecolor=fc, edgecolor=ec)

        # plot the chosen one
        ps = 100
        fc = 'white'
        ax1.scatter(x[iind], y[iind], s=ps, lw=lw,
                facecolor=fc, edgecolor=ec)
        ax2.scatter(x[iind], y[iind], s=ps, lw=lw, 
                facecolor=fc, edgecolor=ec)
        ax3.scatter(x[iind], y[iind], s=ps, lw=lw,
                facecolor=fc, edgecolor=ec)


        # plot central (and the ones you drew anyway)
        fc = 'black'
        ax1.scatter(x[jind], y[jind], s=ps, lw=lw,
                facecolor=fc, edgecolor=ec)
        ax2.scatter(x[jind], y[jind], s=ps, lw=lw, 
                facecolor=fc, edgecolor=ec)
        ax3.scatter(x[jind], y[jind], s=ps, lw=lw,
                facecolor=fc, edgecolor=ec)



        ax1.set_xlim((lowlim2,uplim2))
        ax1.set_ylim((lowlim2,uplim2))
        ax2.set_xlim((lowlim2,uplim2))
        ax2.set_ylim((lowlim2,uplim2))
        ax3.set_xlim((lowlim2,uplim2))
        ax3.set_ylim((lowlim2,uplim2))



        ax1.set_title(r'$x$ component of $\mathbf{A}_{ij}$')
        ax2.set_title(r'$y$ component of $\mathbf{A}_{ij}$')
        ax3.set_title(r'$|\mathbf{A}_{ij}|$')
        ax1.set_xlabel('x')
        ax1.set_ylabel(kernel + ' kernel', fontsize=14)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')


    fig.suptitle(r"Integrand of effective Area $\mathbf{A}_{ij}(\mathbf{x}) = \psi_i(\mathbf{x}) \nabla \psi_j(\mathbf{x}) - \psi_j(\mathbf{x}) \nabla \psi_i(\mathbf{x})$ of a particle (white) ""\n"r" w.r.t. the central particle (black) in a uniform distribution for different kernels")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig('effective_area_A_of_x_uniform_different_kernels.png', dpi=300)

    print('finished.')

    return





if __name__ == '__main__':
    main()

