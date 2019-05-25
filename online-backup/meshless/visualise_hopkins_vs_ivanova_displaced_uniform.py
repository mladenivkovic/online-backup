#!/usr/bin/env python3


#===============================================================
# Visualize effectife area for a chosen particle at various
# positions within an otherwise uniform grid with respect to
# the particle at (0.5, 0.5).
#===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size 


import meshless as ms



#---------------------------
# initialize variables
#---------------------------


# temp during rewriting
srcfile = './snapshot_0000.hdf5'    # swift output file
ptype = 'PartType0'                 # for which particle type to look for
pcoord = [0.5, 0.5]                 # coordinates of particle to work for

L = 10

# border limits for plots
lowlim = 0.4
uplim = 0.6
tol = 1e-3 # tolerance for float comparison





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

    ii = 0
    jj = 0
    for i in range(1, filenummax+1, fileskip):
        for j in range(1, filenummax+1, fileskip):

            srcfile = 'snapshot-'+str(i).zfill(3)+'-'+str(j).zfill(3)+'_0000.hdf5'
            print('working for ', srcfile)

            x, y, h, rho, m, ids, npart= ms.read_file(srcfile, ptype)

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

    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(231, aspect='equal')
    ax2 = fig.add_subplot(232, aspect='equal')
    ax3 = fig.add_subplot(233, aspect='equal')
    ax4 = fig.add_subplot(234, aspect='equal')
    ax5 = fig.add_subplot(235, aspect='equal')
    ax6 = fig.add_subplot(236, aspect='equal')


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

    # reset lowlim and maxlim so cells are centered around the point they represent
    dx = (uplim - lowlim) / AH.shape[0]


    #  uplim2 = uplim - 0.005
    #  lowlim2 = lowlim + 0.005
    uplim2 = uplim
    lowlim2 = lowlim


    cmap = 'YlGnBu_r'

    im1 = ax1.imshow(AHx, origin='lower', 
            vmin=Hxmin, vmax=Hxmax, cmap=cmap,
            extent=(lowlim2, uplim2, lowlim2, uplim2))
    im2 = ax2.imshow(AHy, origin='lower', 
            vmin=Hymin, vmax=Hymax, cmap=cmap,
            extent=(lowlim2, uplim2, lowlim2, uplim2))
    im3 = ax3.imshow(AHnorm, origin='lower', 
            vmin=Hnormmin, vmax=Hnormmax, cmap=cmap,
            extent=(lowlim2, uplim2, lowlim2, uplim2))
    im4 = ax4.imshow(AIx, origin='lower', 
            vmin=Ixmin, vmax=Ixmax, cmap=cmap,
            extent=(lowlim2, uplim2, lowlim2, uplim2))
    im5 = ax5.imshow(AIy, origin='lower', 
            vmin=Iymin, vmax=Iymax, cmap=cmap,
            extent=(lowlim2, uplim2, lowlim2, uplim2))
    im6 = ax6.imshow(AInorm, origin='lower', 
            vmin=Inormmin, vmax=Inormmax, cmap=cmap,
            extent=(lowlim2, uplim2, lowlim2, uplim2))

    for ax, im in [(ax1, im1), (ax2, im2), (ax3, im3), (ax4, im4), (ax5, im5), (ax6, im6)]:
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

    allaxes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in allaxes:
        ax.scatter(xp[mask], yp[mask], s=ps, lw=lw,
            facecolor=fc, edgecolor=ec)

        ax.set_xlim((lowlim2,uplim2))
        ax.set_ylim((lowlim2,uplim2))



    ax1.set_title(r'$x$ component of $\mathbf{A}_{ij}$')
    ax2.set_title(r'$y$ component of $\mathbf{A}_{ij}$')
    ax3.set_title(r'$|\mathbf{A}_{ij}|$')
    ax4.set_title(r'$x$ component of $\mathbf{A}_{ij}$')
    ax5.set_title(r'$y$ component of $\mathbf{A}_{ij}$')
    ax6.set_title(r'$|\mathbf{A}_{ij}|$')

    ax1.set_xlabel('x')
    ax1.set_ylabel('HOPKINS', fontsize=16)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    ax4.set_xlabel('x')
    ax4.set_ylabel('IVANOVA', fontsize=16)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')



    fig.suptitle(r'Effective Area $\mathbf{A}_{ij}$ of a particle w.r.t. the central partilce in a uniform distribution')
    plt.tight_layout()
    plt.savefig('effective_area_displaced_particle.png', dpi=200)

    print('finished.')

    return



    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)

    # convert H to h
    #  H = h
    H = ms.get_H(h)
    pind = ms.find_index(x, y, pcoord)
    nbors = ms.find_neighbours(pind, x, y, H)

    print("Computing effective surfaces")


    A_ij_Hopkins = ms.Aij_Hopkins(pind, x, y, H, m, rho)
    #  A_ij_Ivanova = A_ij_Hopkins
    A_ij_Ivanova = ms.Aij_Ivanova(pind, x, y, H, m, rho)

    x_ij = ms.x_ij(pind, x, y, H, nbors=nbors)

    print("Sum Hopkins:", np.sum(A_ij_Hopkins, axis=0)) 
    print("Sum Ivanova:", np.sum(A_ij_Ivanova, axis=0)) 


    print("Plotting")

    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    pointsize = 200
    ax1.set_facecolor('lavender')
    ax1.scatter(x[pind], y[pind], c='k', s=pointsize*2)
    ax1.set_xlim((0.25,0.75))
    ax1.set_ylim((0.25,0.75))

    ax2.set_facecolor('lavender')
    ax2.scatter(x[pind], y[pind], c='k', s=pointsize*2)
    ax2.set_xlim((0.25,0.75))
    ax2.set_ylim((0.25,0.75))

    # plot particles in order of distance:
    # closer ones first, so that you still can see the short arrows

    dist = np.zeros(len(nbors))
    for i, n in enumerate(nbors):
        dist[i] = (x[n]-pcoord[0])**2 + (y[n]-pcoord[1])**2

    args = np.argsort(dist)

    for i in range(len(nbors)):

        ii = args[i]
        n = nbors[ii]

        cc = i
        while cc > ncolrs-1:
            cc -= ncolrs
        col = fullcolorlist[cc]

        arrwidth = 2

        ax1.scatter(x[n], y[n], c=col, s=pointsize, zorder=0, lw=1, edgecolor='k')
        ax1.arrow(x_ij[ii][0], x_ij[ii][1], A_ij_Hopkins[ii][0], A_ij_Hopkins[ii][1], 
            color=col, lw=arrwidth, zorder=10+i)

        ax2.scatter(x[n], y[n], c=col, s=pointsize, zorder=0, lw=1, edgecolor='k')
        ax2.arrow(x_ij[ii][0], x_ij[ii][1], A_ij_Ivanova[ii][0], A_ij_Ivanova[ii][1], 
            color=col, lw=arrwidth, zorder=10+i)



    ax1.set_title(r'Hopkins $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$', fontsize=18, pad=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.set_title(r'Ivanova $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$', fontsize=18, pad=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')


    plt.tight_layout()
    plt.savefig('effective_area_hopkins_vs_ivanova.png', dpi=200)






if __name__ == '__main__':
    main()

