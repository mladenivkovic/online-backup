#!/usr/bin/env python3


# ===============================================================
# Visualize the effective area from a uniform distribution
# where the smoothing lengths have been computed properly.
# This program is not written flexibly, and will only do the
# plots for one specific particle of this specific test case.
#
# For a chosen particle, for each neighbour within H the
# effective surface is plotted as a vectors in the plane
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt


import meshless as ms


# ---------------------------
# initialize variables
# ---------------------------


# temp during rewriting
srcfile = "./snapshot_0000.hdf5"  # swift output file
ptype = "PartType0"  # for which particle type to look for
pcoord = [0.5, 0.5]  # coordinates of particle to work for
pind = None  # index of particle you chose with pcoord
npart = 0

nbors = []  # indices of all relevant neighbour particles


fullcolorlist = [
    "red",
    "green",
    "blue",
    "gold",
    "magenta",
    "cyan",
    "lime",
    "saddlebrown",
    "darkolivegreen",
    "cornflowerblue",
    "orange",
    "dimgrey",
    "navajowhite",
    "darkslategray",
    "mediumpurple",
    "lightpink",
    "mediumseagreen",
    "maroon",
    "midnightblue",
    "silver",
]

ncolrs = len(fullcolorlist)


def main():

    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)
    pind = ms.find_index(x, y, pcoord)
    H = ms.get_H(h)
    nbors = ms.find_neighbours(pind, x, y, H)

    print("Computing effective surfaces")

    A_ij = ms.Aij_Hopkins(pind, x, y, H, m, rho)
    x_ij = ms.x_ij(pind, x, y, H, nbors=nbors)

    print("Plotting")

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111)

    pointsize = 200
    ax1.set_facecolor("lavender")
    ax1.scatter(x[pind], y[pind], c="k", s=pointsize * 2)
    ax1.set_xlim((0.25, 0.75))
    ax1.set_ylim((0.25, 0.75))

    for i, n in enumerate(nbors):
        cc = i
        while cc > ncolrs:
            cc -= ncolrs
        col = fullcolorlist[cc]

        ax1.scatter(x[n], y[n], c=col, s=pointsize, zorder=0, lw=1, edgecolor="k")
        #  arrind = int(((x[pind]-x[n])**2+(y[pind]-y[n])**2)/(2*0.1*0.1)+1)
        #  arrwidth = arrind*2
        arrind = 2
        arrwidth = arrind * 2
        ax1.arrow(
            x_ij[i][0],
            x_ij[i][1],
            A_ij[i][0],
            A_ij[i][1],
            color=col,
            lw=arrwidth,
            zorder=100 - arrind,
        )

    ax1.set_title(
        r"Hopkins $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$"
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    plt.savefig("effective_area_all_neighbours.png", dpi=200)


if __name__ == "__main__":
    main()
