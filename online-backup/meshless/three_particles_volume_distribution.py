#!/usr/bin/env python3

# Colour in the particle weights of 3 particles
# at points in a plane by computing psi at every x
# and assigning R, G or B values according to the
# value of the psis.
# Do this for different smoothing lengths h.
# all this in a [0x1] x [0x1] box.



import numpy as np
from matplotlib import pyplot as plt

import meshless as ms


#  smoothing_lengths = [   0.25, 0.2,  0.15,
#                          0.1,  0.05, 0.01]
smoothing_lengths = [   2,      1,  0.5,
                        0.25, 0.2,  0.15,
                        0.1,  0.05, 0.01]
nrows = 3
ncols = 3
nx = 10    # how many points to compute for

# particle positions
p1 = [0.2, 0.2]
p2 = [0.4, 0.8]
p3 = [0.7, 0.4]


#  kernels = ['gaussian']
#  kernels = ['cubic_spline']
kernels = ['cubic_spline', 'gaussian']



#===================================
def compute_psis(x, y, h, kernel):
#===================================

    psis = np.zeros(3, dtype=np.float128)

    for i, part in enumerate([p1, p2, p3]):
        psis[i] = ms.psi(x, y, part[0], part[1], h, kernel)

    if np.sum(psis)==0:
        psis = 0
    else:
        psis /= np.sum(psis)

    return psis






#==============================
def compute_image(h, kernel):
#==============================
    """
    Computes the image by computing the psi of every particle at every position.
    """

    image = np.zeros((nx, nx, 3), dtype=np.float64)

    dx = 1./nx

    for i in range(nx):
        x = (i+0.5)*dx
        for j in range(nx):
            y = (j+0.5)*dx

            image[j, i] = compute_psis(x, y, h, kernel)

    return image





#==============================
if __name__ == '__main__':
#==============================

    for kernel in kernels:
        print("working for", kernel, "kernel")

        # create subplots
        fig = plt.figure(figsize = (10, 10))
        axes = [None for h in smoothing_lengths]

        i = 0
        for r in range(nrows):
            for c in range(ncols):
                axes[i] = fig.add_subplot(nrows, ncols, i+1)
                i+=1


        for i, h in enumerate(smoothing_lengths):
            print("Working for h=", h)
            image = compute_image(h, kernel)
            axes[i].imshow(image, 
                    origin='lower',
                    extent=(0,1,0,1))

            axes[i].set_title('h = '+str(h))

            for part, col in [(p1,'red'), (p2, 'green'), (p3, 'blue')]:
                axes[i].scatter(part[0], part[1], c=col, s=60, edgecolor='k', lw=2)

        fig.suptitle(r'$\psi(x)$ for three particles with a '+kernel+' kernel')
        plt.tight_layout(rect=(0,0,1,0.95))

        plt.savefig('psi_x_for_three_particles-'+kernel+'.png', dpi=300)



