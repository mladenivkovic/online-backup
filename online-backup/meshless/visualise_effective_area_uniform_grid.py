#!/usr/bin/env python3


#===============================================================
# Visualize the effective area from a uniform distribution
# where the smoothing lengths have been computed properly.
# This program is not written flexibly, and will only do the
# plots for one specific particle of this specific test case.
#
# For a chosen particle, for each neighbour within 2h the 
# effective surface is plotted as a vectors in the plane
#===============================================================


import numpy as np
import matplotlib.pyplot as plt

import h5py


x = None
y = None
h = None
rho = None
m = None
srcfile = './snapshot_0000.hdf5'    # swift output file
ptype = 'PartType0'                 # for which particle type to look for
pcoord = [0.5, 0.5]                 # coordinates of particle to work for
pind = None                         # index of particle you chose with pcoord
npart = 0

nbors = []                          # indices of all relevant neighbour particles



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



#=========================
def read_file():
#=========================
    """
    Just read the file man.
    """

    f = h5py.File(srcfile)

    global x, y, h, npart, rho, m

    x = f[ptype]['Coordinates'][:,0]
    y = f[ptype]['Coordinates'][:,1]
    h = f[ptype]['SmoothingLength'][:]
    rho = f[ptype]['Density'][:]
    m = f[ptype]['Masses'][:]

    npart = x.shape[0]

    f.close()

    return




#========================
def find_index():
#========================
    """
    Find the index in the read-in arrays where
    the particle with coordinates of your choice is
    """

    global pind

    for i in range(npart):
        if x[i]==pcoord[0] and y[i] == pcoord[1]:
            pind = i
            break

    print("got pind", pind, 'with x=', x[pind], 'y=', y[pind], 'h=', h[pind])

    return




#=============================
def find_neighbours(ind):
#=============================
    """
    Find indices of all neighbours within 2h (where kernel != 0)
    """


    x0 = x[ind]
    y0 = y[ind]
    fhsq = h[ind]*h[ind]*4
    neigh = []

    for i in range(x.shape[0]):
        if i==ind:
            continue

        dist = (x[i]-x0)**2 + (y[i]-y0)**2
        if dist < fhsq:
            neigh.append(i)

    return neigh




#==================
def W(q):
#==================
    """
    cubic spline kernel
    """ 
    sigma = 1./np.pi
    if q < 0.5:
        return 1. - q*q * (1.5 - 0.75*q) 
    elif q < 2:
        return 0.25*(2-q)**3
    else:
        return 0



#===============
def V(ind):
#===============
    """
    Volume estimate for particle with index ind
    """

    return m[ind]/rho[ind]




#=======================
def psi(x, y, xi, yi, h):
#=======================
    """
    UNNORMALIZED Volume fraction at position x of some particle part
    ind: neighbour index in x/y/h array
    """
    q = np.float128(np.sqrt((x - xi)**2 + (y - yi)**2)/h)

    return W(q)



#=============================================
def get_matrix(xi, yi, xj, yj, psi_j):
#=============================================
    """
    Get B_i ^{alpha beta}
    xi, yi: floats; Evaluate B at this position
    xj, yj: arrays; Neighbouring points
    psi_j:  array;  volume fraction of neighbours at position x_i; psi_j(x_i)
    """

    E00 = np.sum((xj-xi)**2 * psi_j)
    E01 = np.sum((xj-xi)*(yj-yi) * psi_j)
    E11 = np.sum((yj-yi)**2 * psi_j)
          
    E = np.matrix([[E00, E01], [E01, E11]])

    B = E.getI()
    return B



#=============================================
def compute_psi(xi, yi, xj, yj, h):
#=============================================
    """
    Compute all psi_j(x_i)
    xi, yi: floats
    xj, yj: arrays
    h: float
    """

    # psi_j(x_i)
    psi_j = np.zeros(xj.shape[0], dtype=np.float128)

    for i in range(xj.shape[0]):
        psi_j[i] = psi(xi, yi, xj[i], yj[i], h)

    return psi_j






#================================
def get_effective_surfaces():
#================================
    """
    Compute and plot the effective area using proper gradients
    but only at "midpoints"
    """

    print("Computing effective surfaces")

    xj = x[nbors]
    yj = y[nbors]
    hj = h[nbors]



    #-------------------------------------------------------
    # Part 1: For particle at x_i (Our chosen particle)
    #-------------------------------------------------------

    # compute psi_j(x_i)
    psi_j = compute_psi(x[pind], y[pind], xj, yj, h[pind])

    # normalize psi_j
    omega_xi =  (np.sum(psi_j) + psi(x[pind], y[pind], x[pind], y[pind], h[pind]))
    psi_j /= omega_xi
    psi_j = np.float64(psi_j)

    # compute B_i
    B_i = get_matrix(x[pind], y[pind], xj, yj, psi_j)

    # compute grad_psi_j(x_i)
    grad_psi_j = np.empty((len(nbors), 2), dtype=np.float)
    for i, n in enumerate(nbors):
        dx = np.array([xj[i]-x[pind], yj[i]-y[pind]])
        grad_psi_j[i] = np.dot(B_i, dx) * psi_j[i]



    #---------------------------------------------------------------------------
    # Part 2: values of psi/grad_psi of particle i at neighbour positions x_j
    #---------------------------------------------------------------------------

    psi_i = np.zeros(len(nbors), dtype=np.float128)            # psi_i(xj)
    grad_psi_i = np.empty((len(nbors), 2), dtype=np.float)  # grad_psi_i(x_j)

    for i,n in enumerate(nbors):
        # first compute all psi(xj) from neighbour's neighbours to get weight omega
        nneigh = find_neighbours(n)
        xk = x[nneigh]
        yk = y[nneigh]
        for j, nn in enumerate(nneigh):
            psi_k = compute_psi(x[n], y[n], xk, yk, h[n])
            if nn == pind: # store psi_i, which is the psi for the particle whe chose at position xj; psi_i(xj)
                psi_i[i] = psi_k[j]
    
        omega_xj = (np.sum(psi_k) + psi(x[n], y[n], x[n], y[n], h[n]))

        psi_i[i]/= omega_xj


        # now compute B_j^{\alpha \beta}
        B_j = get_matrix(x[n], y[n], xk, yk, h[n])

        # get gradient
        dx = np.array([x[pind]-x[n], y[pind]-y[n]])
        grad_psi_i[i] = np.dot(B_j, dx) * np.float64(psi_i[i])



    #-------------------------------
    # Part 3: Compute A_ij, x_ij
    #-------------------------------

    A_ij = np.empty((len(nbors),2), dtype = np.float)
    x_ij = np.empty((len(nbors),2), dtype = np.float)

    for i,n in enumerate(nbors):
        A_ij[i] = V(pind)*grad_psi_j[i] - V(n)*grad_psi_i[i]

        hfact = h[pind]/(h[pind]+h[n])
        x_ij[i] = np.array([x[pind]-hfact*(x[pind]-x[n]), y[pind]-hfact*(y[pind]-y[n])])




    #---------------------------
    # Part 4: Plot results
    #---------------------------

    print("Plotting")

    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)

    pointsize = 200
    ax1.set_facecolor('lavender')
    ax1.scatter(x[pind], y[pind], c='k', s=pointsize*2)
    ax1.set_xlim((0.25,0.75))
    ax1.set_ylim((0.25,0.75))

    for i,n in enumerate(nbors):
        cc = i
        while cc > ncolrs:
            cc -= ncolrs
        col = fullcolorlist[cc]

        ax1.scatter(x[n], y[n], c=col, s=pointsize, zorder=0, lw=1, edgecolor='k')
        #  arrind = int(((x[pind]-x[n])**2+(y[pind]-y[n])**2)/(2*0.1*0.1)+1)
        #  arrwidth = arrind*2
        arrind = 2
        arrwidth = arrind*2
        ax1.arrow(x_ij[i][0], x_ij[i][1], A_ij[i][0], A_ij[i][1], 
            color=col, lw=arrwidth, zorder=100-arrind)



    ax1.set_title(r'$\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')


    plt.savefig('effective_area_all_neighbours.png', dpi=200)

    return




#========================
def main():
#========================
    

    global nbors
    read_file()
    find_index()
    nbors = find_neighbours(pind)

    get_effective_surfaces()

    





if __name__ == '__main__':
    main()

