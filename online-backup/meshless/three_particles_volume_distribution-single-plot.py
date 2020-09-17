#!/usr/bin/env python3

# Colour in the particle weights of 3 particles
# at points in a plane by computing psi at every x
# and assigning R, G or B values according to the
# value of the psis.
# Do this for different smoothing lengths h.
# all this in a [0x1] x [0x1] box.


import numpy as np
from matplotlib import pyplot as plt

import astro_meshless_surfaces as ml
from my_utils import setplotparams_single_plot

setplotparams_single_plot(for_presentation=True)

nrows = 1
ncols = 1
nx = 300  # how many points to compute for

# particle positions
p1 = np.array([0.2, 0.2])
p2 = np.array([0.4, 0.8])
p3 = np.array([0.7, 0.4])

h = 1.1
L = np.array([1.0, 1.0])


def compute_psis(x, y):

    psis = np.zeros(3, dtype=np.float128)

    for i, part in enumerate([p1, p2, p3]):
        dx, dy = ml.get_dx(x, part[0], y, part[1], L=L, periodic=True)
        q = np.sqrt(dx ** 2 + dy ** 2) / h
        psis[i] = ml.W(q, h)

    if np.sum(psis) == 0:
        psis = 0
    else:
        psis /= np.sum(psis)

    return psis


def compute_image():
    """
    Computes the image by computing the psi of every particle at every position.
    """

    image = np.zeros((nx, nx, 3), dtype=np.float64)

    dx = 1.0 / nx

    for i in range(nx):
        x = (i + 0.5) * dx
        for j in range(nx):
            y = (j + 0.5) * dx

            image[j, i] = compute_psis(x, y)

    return image


if __name__ == "__main__":

    # create subplots
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    image = compute_image()
    ax.imshow(image, origin="lower", extent=(0, 1, 0, 1))

    for part, col in [(p1, "red"), (p2, "green"), (p3, "blue")]:
        ax.scatter(part[0], part[1], c=col, s=60, edgecolor="k", lw=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(r"volume distribution between three particles")

    plt.savefig("volume_distribution_three_particles.png")
