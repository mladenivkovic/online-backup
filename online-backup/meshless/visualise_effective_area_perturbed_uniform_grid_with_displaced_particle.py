#!/usr/bin/env python3


#===============================================================
# Visualize effectife area for a chosen particle at various
# positions within an otherwise uniform grid with respect to
# the particle at (0.5, 0.5).
#===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import h5py
import meshless as ms


ptype = 'PartType0'             # for which particle type to look for
pind = -1                       # index of particle you chose with pcoord
cind = None                     # index of particle in the center (0.5, 0.5)


L = 10      # nr of particles along one axis
boxSize = 1

# border limits for plots
lowlim = 0.4-0.2*boxSize/L
uplim = 0.4-(0.2-203/100)*boxSize/L

tol = 1e-3 # tolerance for float comparison

cheatfact = 1.5 # enlarge H with this cheat to skip singular matrices







#========================
def main():
#========================


    #-----------------------------
    # Part1 : compute all A
    #-----------------------------

    print("Computing effective surfaces")

    nx, filenummax, fileskip = ms.get_sample_size()

    A = np.zeros((nx, nx, 2), dtype=np.float) # storing computed effective surfaces

    ii = 0
    jj = 0
    for i in range(1, filenummax+1, fileskip):
        for j in range(1, filenummax+1, fileskip):

            srcfile = 'snapshot-'+str(i).zfill(3)+'-'+str(j).zfill(3)+'_0000.hdf5'
            print('working for ', srcfile)

            x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)

            H = ms.get_H(h)
            H *= cheatfact

            cind = ms.find_central_particle(L, ids)
            pind = ms.find_added_particle(ids)

            nbors = ms.find_neighbours(pind, x, y, H)
            Aij = ms.Aij_Hopkins(pind, x, y, H, m, rho)

            try:
                ind = nbors.index(cind)
            except ValueError:
                print(nbors)
                print(x[nbors])
                print(y[nbors])
                print(cind, x[cind], y[cind])
                print(pind, x[pind], y[pind], 2*h[pind])
            A[jj, ii] = Aij[ind]

            jj += 1

        ii += 1
        jj = 0





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
    xp = x[inds][:-1]
    yp = y[inds][:-1]

    mask = np.logical_and(xp>=lowlim-tol, xp<=uplim+tol)
    mask = np.logical_and(mask, yp>=lowlim-tol)
    mask = np.logical_and(mask, yp<=uplim+tol)

    ps = 100
    fc = 'white'
    ec = 'black'
    lw = 2

    # scatter neighbours
    ax1.scatter(xp[mask], yp[mask], s=ps, lw=lw,
            facecolor=fc, edgecolor=ec)
    ax2.scatter(xp[mask], yp[mask], s=ps, lw=lw,
            facecolor=fc, edgecolor=ec)
    ax3.scatter(xp[mask], yp[mask], s=ps, lw=lw,
            facecolor=fc, edgecolor=ec)


    # scatter central
    ax1.scatter(x[cind], y[cind], s=ps, lw=lw,
            facecolor=ec, edgecolor=ec)
    ax2.scatter(x[cind], y[cind], s=ps, lw=lw,
            facecolor=ec, edgecolor=ec)
    ax3.scatter(x[cind], y[cind], s=ps, lw=lw,
            facecolor=ec, edgecolor=ec)

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


    fig.suptitle(r'Effective Area $\mathbf{A}_{ij}$ of a particle w.r.t. the central particle (black) in a perturbed uniform distribution')
    plt.tight_layout()
    plt.savefig('effective_area_perturbed_uniform_displaced_particle.png', dpi=300)

    print('finished.')

    return





if __name__ == '__main__':
    main()

