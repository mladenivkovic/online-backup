#!/usr/bin/env python3


#===============================================================
# Visualize the effective area from a uniform distribution
# where the smoothing lengths have been computed properly.
# This program is not written flexibly, and will only do the
# plots for one specific particle of this specific test case.
#===============================================================


import numpy as np
import matplotlib.pyplot as plt


import meshless as ms



#---------------------------
# initialize variables
#---------------------------


# temp during rewriting
srcfile = './snapshot_0000.hdf5'    # swift output file
ptype = 'PartType0'                 # for which particle type to look for
pcoord = [0.5, 0.5]                 # coordinates of particle to work for




fullcolorlist=['red', 
        'green', 
        'blue', 
        'gold', 
        'magenta', 
        'cyan',
        'lime',
        'saddlebrown',
        'darkolivegreen',
        'cornflowerblue',
        'orange',
        'dimgrey',
        'navajowhite',
        'darkslategray',
        'mediumpurple',
        'lightpink',
        'mediumseagreen',
        'maroon',
        'midnightblue',
        'silver']

ncolrs = len(fullcolorlist)





#========================
def main():
#========================
    

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


    print("IVANOVA:", A_ij_Ivanova)
    print("HOPKINS:", A_ij_Hopkins)


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

