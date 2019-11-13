#!/usr/bin/env python3


#===============================================================
# 2 particles, uniformly distributed in a box for manual
# testing purposes
#===============================================================


import numpy as np
import matplotlib.pyplot as plt


import meshless as ms
import my_utils
my_utils.setplotparams_multiple_plots()



#---------------------------
# initialize variables
#---------------------------

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


    #  x = np.array([0.25, 0.25, 0.75, 0.75])
    #  y = np.array([0.25, 0.75, 0.25, 0.75])
    #  h = np.array([0.50, 0.50, 0.50, 0.50])
    #  m = np.array([0.25, 0.25, 0.25, 0.25])
    #  rho = np.array([1.00, 1.00, 1.00, 1.00])
    #  npart = x.shape[0]
    #  ids = np.arange(npart, dtype=np.int)
    x = np.array([0.25, 0.75])
    y = np.array([0.25, 0.75])
    h = np.array([1.00, 1.00])
    m = np.array([0.50, 0.50])
    rho = np.array([1.00, 1.00])
    npart = x.shape[0]
    ids = np.array([0, 1])


    pind = 0
    j = ms.find_neighbours(pind, x, y, h)

    A_ij_I = ms.Aij_Ivanova(pind, x, y, h, m, rho)
    print("For 2 particles, the matrix becomes singular and the code crashes here.")
    A_ij_H = ms.Aij_Hopkins_v2(pind, x, y, h, m, rho)

    x_ij = ms.x_ij(pind, x, y, h, nbors=j)



    print("Plotting")

    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    pointsize = 200
    ax1.set_facecolor('lavender')
    ax1.scatter(x[pind], y[pind], c='k', s=pointsize*2)

    ax2.set_facecolor('lavender')
    ax2.scatter(x[pind], y[pind], c='k', s=pointsize*2)

    # plot particles in order of distance:
    # closer ones first, so that you still can see the short arrows

    nbors = j
    dist = np.zeros(len(nbors))
    for i, n in enumerate(nbors):
        dist[i] = (x[n]-x[pind])**2 + (y[n]-y[pind])**2

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
        ax1.arrow(x_ij[ii][0], x_ij[ii][1], A_ij_H[ii][0], A_ij_H[ii][1],
            color=col, lw=arrwidth, zorder=10+i)

        ax2.scatter(x[n], y[n], c=col, s=pointsize, zorder=0, lw=1, edgecolor='k')
        ax2.arrow(x_ij[ii][0], x_ij[ii][1], A_ij_I[ii][0], A_ij_I[ii][1],
            color=col, lw=arrwidth, zorder=10+i)



    ax1.set_title(r'Hopkins $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$') #, fontsize=18, pad=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.set_title(r'Ivanova $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$')#, fontsize=18, pad=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)


    plt.savefig('2_particle_situation.png')



if __name__ == '__main__':
    main()

