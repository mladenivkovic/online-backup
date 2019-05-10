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


ptype = 'PartType0'             # for which particle type to look for
pind = -1                       # index of particle you chose with pcoord
cind = None                     # index of particle in the center (0.5, 0.5)


L = 10      # nr of particles along one axis
boxSize = 1

# border limits for plots
lowlim = 0.4-0.2*boxSize/L
uplim = 0.4-(0.2-203/100)*boxSize/L

print("lowlim:", lowlim, "uplim:", uplim)
tol = 1e-3 # tolerance for float comparison




#=========================
def read_file(srcfile):
#=========================
    """
    Just read the file man.
    """

    f = h5py.File(srcfile)


    x = f[ptype]['Coordinates'][:,0]
    y = f[ptype]['Coordinates'][:,1]
    h = f[ptype]['SmoothingLength'][:]
    rho = f[ptype]['Density'][:]
    m = f[ptype]['Masses'][:]

    ids = f[ptype]['ParticleIDs'][:]

    global pind, cind

    L = int(np.sqrt(x.shape[0]-1)+0.5)

    # compute ID of central particle at ~(0.5, 0.5)
    i = L//2-1
    cid = i*L + i + 1
    cind = np.asscalar(np.where(ids==cid)[0])

    # get ID of added particle
    pid = x.shape[0]
    pind = np.asscalar(np.where(ids==pid)[0])

    f.close()

    return x, y, h, rho, m, ids





#======================================
def find_neighbours(ind, x, y, h):
#======================================
    """
    Find indices of all neighbours within 2h (where kernel != 0) for particle with index ind
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
def W(q, h):
#==================
    """
    cubic spline kernel
    """ 
    sigma = np.float128(10./(7*np.pi*h**2))
    if q < 1:
        return np.float128(1. - q*q * (1.5 - 0.75*q))
    elif q < 2:
        return np.float128(0.25*(2-q)**3)
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

    return W(q, h)



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






#==================================================
def get_effective_surface(x, y, h, rho, m, nbors):
#==================================================
    """
    Compute and plot the effective area using proper gradients
    """

    xj = x[nbors]
    yj = y[nbors]
    hj = h[nbors]

    #-------------------------------------------------------
    # Part 1: For particle at x_i (Our chosen particle)
    #-------------------------------------------------------

    # compute psi_j(x_i)
    psi_j = compute_psi(x[pind], y[pind], xj, yj, h[pind])

    # normalize psi_j. Don't forget to add the self-contributing value!
    omega_xi =  (np.sum(psi_j) + psi(x[pind], y[pind], x[pind], y[pind], h[pind]))
    psi_j /= omega_xi
    psi_j = np.float64(psi_j)

    # compute B_i
    B_i = get_matrix(x[pind], y[pind], xj, yj, psi_j)

    # get index of central particle in nbors list
    cind_n = nbors.index(cind)

    # compute grad_psi_j(x_i)
    grad_psi_j = np.empty((1, 2), dtype=np.float)
    dx = np.array([x[cind]-x[pind], y[cind]-y[pind]])
    grad_psi_j = np.dot(B_i, dx) * psi_j[cind_n]



    #---------------------------------------------------------------------------
    # Part 2: values of psi/grad_psi of particle i at neighbour positions x_j
    #---------------------------------------------------------------------------

    psi_i = 0.0                                    # psi_i(xj)
    grad_psi_i = np.empty((1, 2), dtype=np.float)  # grad_psi_i(x_j)

    # first compute all psi(xj) from central's neighbours to get weight omega
    nneigh = find_neighbours(cind, x, y, h)
    xk = x[nneigh]
    yk = y[nneigh]
    for j, nn in enumerate(nneigh):
        psi_k = compute_psi(x[cind], y[cind], xk, yk, h[cind])
        if nn == pind: # store psi_i, which is the psi for the particle whe chose at position xj; psi_i(xj)
            psi_i = psi_k[j]

    omega_xj = (np.sum(psi_k) + psi(x[cind], y[cind], x[cind], y[cind], h[cind]))

    psi_i/= omega_xj
    psi_i = np.float64(psi_i)


    # now compute B_j^{\alpha \beta}
    B_j = get_matrix(x[cind], y[cind], xk, yk, h[cind])

    # get gradient
    dx = np.array([x[pind]-x[cind], y[pind]-y[cind]])
    grad_psi_i = np.dot(B_j, dx) * psi_i



    #-------------------------------
    # Part 3: Compute A_ij
    #-------------------------------

    V = m/rho

    A_ij = None

    A_ij = V[pind]*grad_psi_j - V[cind]*grad_psi_i

    if A_ij is None:
        print("PROBLEM: A_IJ IS NONE")
        raise ValueError
    else:
        return A_ij











#========================
def main():
#========================
    

    #-----------------------------
    # Part1 : compute all A
    #-----------------------------

    print("Computing effective surfaces")

    A = np.zeros((102, 102, 2), dtype=np.float) # storing computed effective surfaces
    #  A = np.zeros((10, 10, 2), dtype=np.float) # storing computed effective surfaces

    global pind

    ii = 0
    jj = 0
    for i in range(1, 204, 2):
        for j in range(1, 204, 2):
    #  for i in range(1, 200, 20):
    #      for j in range(1, 200, 20):

            srcfile = 'snapshot-'+str(i).zfill(3)+'-'+str(j).zfill(3)+'_0000.hdf5'
            print('working for ', srcfile)

            x, y, h, rho, m , ids= read_file(srcfile)
            # displaced particle has index -1
            nbors = find_neighbours(pind, x, y, h)

            A[jj, ii] = get_effective_surface(x, y, h, rho, m, nbors)
            
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

