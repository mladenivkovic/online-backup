#!/usr/bin/env python3

#===============================================================
# Compute A(x) between two specified particles at various
# positions x
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
nx = 100
tol = 1e-5 # tolerance for float comparison


x = None
y = None
h = None
rho = None
m = None
ids = None
V = None

# define list of kernels
#  https://pysph.readthedocs.io/en/latest/reference/kernels.html

kernels = ['cubic_spline', 'quintic_spline', 
        'gaussian', 'gaussian_compact', 'supergaussian',
        'wendland_C2', 'wendland_C4', 'wendland_C6']


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
def find_neighbours(xx, yy, hh, kernel):
#======================================
    """
    Find indices of all neighbours within compact support
    from the position (xx, yy) with smoothing length h=hh
    """

    if kernel in ['quintic_spline', 'gaussian_compact', 'supergaussian']:
        fhsq = hh*hh*9
    elif kernel == 'gaussian':
        fhsq = 1e10
    else:
        fhsq = hh*hh*4
    neigh = []

    for i in range(x.shape[0]):
        dist = (x[i]-xx)**2 + (y[i]-yy)**2
        if dist <= fhsq:
            neigh.append(i)

    return neigh




#==================
def W(q, h, kernel):
#==================
    """
    Various kernels
    """ 

    if kernel == 'cubic_spline': 

        sigma = 1./np.pi
        if q < 0.5:
            return 1. - q*q * (1.5 - 0.75*q) 
        elif q < 2:
            return 0.25*(2-q)**3
        else:
            return 0


    elif kernel == 'quintic_spline':

        sigma = 7/(478*np.pi*h*h)
        if q <= 1:
            return sigma * ((3-q)**5 - 6*(2-q)**5 + 15*(1-q)**5)
        elif q<=2:
            return sigma * ((3-q)**5 - 6*(2-q)**5)
        elif q<=3:
            return sigma * ((3-q)**5)
        else:
            return 0



    elif kernel == 'gaussian':
        # gaussian without compact support
        return 1./(np.sqrt(0.5*np.pi)*h)**3*np.exp(-2*q**2)

        

    elif kernel == 'gaussian_compact':
        # gaussian with compact support

        sigma = 1./(np.pi*h*h)

        if q <= 3:
            return sigma * np.exp(-q**2)
        else:
            return 0




    elif kernel == 'supergaussian':


        if q <= 3:
            sigma = 1./(np.sqrt(np.pi)*h)**3
            return sigma * np.exp(-q**2)*(2.5 - q**2)
        else:
            return 0


    elif kernel == 'wendland_C2':

        if q <= 2:
            sigma = 7/(4*np.pi * h**2)
            return sigma * (1 - 0.5*q)**4*(2*q+1)
        else:
            return 0


    elif kernel == 'wendland_C4':

        if q <= 2:
            sigma = 9/(4*np.pi*h**2)
            return sigma*(1-0.5*q)**6 * (35/12*q**2 + 3*q + 1)
        else:
            return 0


    elif kernel == 'wendland_C6':
        
        if q <= 2:
            sigma = 78/(28*np.pi*h**2)
            return sigma * (1 - 0.5*q)**8*(4*q**3 + 6.25*q**2 + 4*q + 1)
        else:
            return 0


    else:
        print("Didn't find kernel", kernel)
        quit()



#====================================
def psi(x, y, xi, yi, h, kernel):
#====================================
    """
    UNNORMALIZED Volume fraction at position x of some particle part
    """
    q = np.float128(np.sqrt((x - xi)**2 + (y - yi)**2)/h)

    return W(q, h, kernel)



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
def compute_psi(xx, yy, xj, yj, hh, kernel):
#=============================================
    """
    Compute all psi_j(x_i)
    xi, yi: floats
    xj, yj: arrays
    h: float
    """

    # psi_j(x_i)
    psis = np.zeros(xj.shape[0], dtype=np.float128)

    for i in range(xj.shape[0]):
        psis[i] = psi(xx, yy, xj[i], yj[i], hh, kernel)

    return psis






#==================================================
def get_effective_surface(xx, yy, hh, nbors, kernel):
#==================================================
    """
    Compute and plot the effective area using proper gradients
    """

    xj = x[nbors]
    yj = y[nbors]
    hj = h[nbors]

    # compute all psi(x)
    psis = compute_psi(xx, yy, xj, yj, hh, kernel)

    # normalize psis
    omega =  (np.sum(psis))
    psis /= omega
    psis = np.float64(psis)


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
    


    #------------------
    # READ IN FILE
    #------------------

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

            for j in range(nx):
                yy = lowlim + dx * j


                hh = get_smoothing_length(xx, yy)

                nbors = find_neighbours(xx, yy, hh, kernel)

                A[j, i] = get_effective_surface(xx, yy, hh, nbors, kernel) # not a typo: need A[j,i] for imshow


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


    fig.suptitle(r'Effective Area $\mathbf{A}_{ij}(\mathbf{x}) = \psi_i(\mathbf{x}) \nabla \psi_j(\mathbf{x}) - \psi_j(\mathbf{x}) \nabla \psi_i(\mathbf{x})$ of a particle (white) w.r.t. the central particle (black) in a uniform distribution for different kernels')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig('effective_area_A_of_x_uniform_different_kernels.png', dpi=300)

    print('finished.')

    return





if __name__ == '__main__':
    main()

