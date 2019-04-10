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
iind = None                     # index of particle at (0.4, 0.4)
jind = None                     # index of particle in the center (0.5, 0.5)


L = 10      # nr of particles along one axis
boxSize = 1

# border limits for plots
lowlim = 0.35
uplim = 0.55
nx = 200
tol = 1e-5 # tolerance for float comparison


x = None
y = None
h = None
rho = None
m = None
ids = None
V = None




#=========================
def read_file(srcfile):
#=========================
    """
    Just read the file man.
    """

    f = h5py.File(srcfile)

    global x, y, h, rho, m, ids, V

    x = f[ptype]['Coordinates'][:,0]
    y = f[ptype]['Coordinates'][:,1]
    h = f[ptype]['SmoothingLength'][:]
    rho = f[ptype]['Density'][:]
    m = f[ptype]['Masses'][:]

    ids = f[ptype]['ParticleIDs'][:]

    V = m/rho

    f.close()

    return 





#======================================
def find_neighbours(xx, yy, hh):
#======================================
    """
    Find indices of all neighbours within 2h (where kernel != 0)
    from the position (xx, yy) with smoothing length h=hh
    """


    fhsq = hh*hh*4
    neigh = []

    for i in range(x.shape[0]):
        dist = (x[i]-xx)**2 + (y[i]-yy)**2
        if dist <= fhsq:
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






#=======================
def psi(x, y, xi, yi, h):
#=======================
    """
    UNNORMALIZED Volume fraction at position x of some particle part
    ind: neighbour index in x/y/h array
    """
    q = np.sqrt((x - xi)**2 + (y - yi)**2)/h

    return W(q)



#=============================================
def get_matrix(xx, yy, xj, yj, psi_j):
#=============================================
    """
    Get B_i ^{alpha beta}
    xx, yy: floats; Evaluate B at this position
    xj, yj: arrays; Neighbouring points
    psi_j:  array;  volume fraction of neighbours at position x; psi(x)
    """

    E00 = np.sum((xj-xx)**2 * psi_j)
    E01 = np.sum((xj-xx)*(yj-yy) * psi_j)
    E11 = np.sum((yj-yy)**2 * psi_j)
          
    E = np.matrix([[E00, E01], [E01, E11]])

    B = E.getI()
    return B



#=============================================
def compute_psi(xx, yy, xj, yj, hh):
#=============================================
    """
    Compute all psi_j(x_i)
    xi, yi: floats
    xj, yj: arrays
    h: float
    """

    # psi_j(x_i)
    psis = np.zeros(xj.shape[0], dtype=np.float)

    for i in range(xj.shape[0]):
        psis[i] = psi(xx, yy, xj[i], yj[i], hh)

    return psis






#==================================================
def get_effective_surface(xx, yy, hh, nbors):
#==================================================
    """
    Compute and plot the effective area using proper gradients
    """

    xj = x[nbors]
    yj = y[nbors]
    hj = h[nbors]

    # compute all psi(x)
    psis = compute_psi(xx, yy, xj, yj, hh)

    # normalize psis
    omega =  (np.sum(psis))
    psis /= omega


    # find where psi_i and psi_j are in that array
    inb = nbors.index(iind)
    jnb = nbors.index(jind)

    # compute matrix B
    B = get_matrix(xx, yy, xj, yj, psis)

    # compute grad_psi_j(x_i)
    dx = np.array([x[iind]-xx, y[iind]-yy])
    grad_psi_i = np.dot(B, dx) * psis[inb]
    dx = np.array([x[jind]-xx, y[jind]-yy])
    grad_psi_j = np.dot(B, dx) * psis[jnb]


    A_ij = psis[inb]*grad_psi_j - psis[jnb]*grad_psi_i

    if A_ij is None:
        print("PROBLEM: A_IJ IS NONE")
        raise ValueError
    else:
        return A_ij









#===================================
def get_smoothing_length(xx, yy):
#===================================
    """
    Compute h(x) at position (xx, yy), where there is 
    not necessariliy a particle
    """
    vol = boxSize*boxSize*boxSize
    hh = np.sum(h*V)/vol
    return hh





#========================
def main():
#========================
    

    #-----------------------------
    # Part1 : compute all A
    #-----------------------------

    print("Computing effective surfaces")

    srcfile = 'snapshot_0000.hdf5'
    read_file(srcfile)

    # find where particles i (0.4, 0.4) and j (0.5, 0.5) are

    global iind, jind
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

        for j in range(nx):
            yy = lowlim + dx * j


            hh = get_smoothing_length(xx, yy)

            nbors = find_neighbours(xx, yy, hh)

            A[j, i] = get_effective_surface(xx, yy, hh,nbors) # not a typo: need A[j,i] for imshow






    #-----------------------------
    # Part2: Plot results
    #-----------------------------

    print("Plotting")

    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(131, aspect='equal')
    ax2 = fig.add_subplot(132, aspect='equal')
    ax3 = fig.add_subplot(133, aspect='equal')


    Ax = A[:,:,0].transpose() # pyplot.imshow takes [y,x] !
    Ay = A[:,:,1].transpose()
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

    cmap = 'jet'

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


    fig.suptitle(r'Effective Area $\mathbf{A}_{ij}(\mathbf{x}) = \psi_i(\mathbf{x}) \nabla \psi_j(\mathbf{x}) - \psi_j(\mathbf{x}) \nabla \psi_i(\mathbf{x})$ of a particle (white) w.r.t. the central particle (black) in a uniform distribution')
    plt.tight_layout()
    plt.savefig('effective_area_A_of_x.png', dpi=300)

    print('finished.')

    return





if __name__ == '__main__':
    main()

