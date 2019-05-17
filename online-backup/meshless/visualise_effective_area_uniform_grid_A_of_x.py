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

L = 10      # nr of particles along one axis
boxSize = 1

# border limits for plots
lowlim = 0.35
uplim = 0.55
nx = 200
tol = 1e-5 # tolerance for float comparison





#========================
def main():
#========================
    

    #-----------------------------
    # Part1 : compute all A
    #-----------------------------

    print("Computing effective surfaces")

    srcfile = 'snapshot_0000.hdf5'
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

    A = np.zeros((nx, nx, 2), dtype=np.float) # storing computed effective surfaces
    dx = (uplim - lowlim)/nx


    for i in range(nx):
        xx = lowlim + dx * i

        print("i =", i+1, "/", nx)

        for j in range(nx):
            yy = lowlim + dx * j

            hh = ms.h_of_x(xx, yy, x, y, h, m, rho)

            A[j, i] = ms.Integrand_Aij_Ivanova(iind, jind, xx, yy, hh, x, y, h, m, rho) # not a typo: need A[j,i] for imshow






    #-----------------------------
    # Part2: Plot results
    #-----------------------------

    print("Plotting")

    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(131, aspect='equal')
    ax2 = fig.add_subplot(132, aspect='equal')
    ax3 = fig.add_subplot(133, aspect='equal')


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
    ax1.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')


    fig.suptitle(r'Integrand of Effective Area $\mathbf{A}_{ij}(\mathbf{x}) = \psi_i(\mathbf{x}) \nabla \psi_j(\mathbf{x}) - \psi_j(\mathbf{x}) \nabla \psi_i(\mathbf{x})$ of a particle (white) w.r.t. the central particle (black) in a uniform distribution')
    plt.tight_layout()
    plt.savefig('effective_area_A_of_x.png', dpi=300)

    print('finished.')

    return





if __name__ == '__main__':
    main()

