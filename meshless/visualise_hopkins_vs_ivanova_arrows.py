#!/usr/bin/env python3


# ===============================================================
# Visualize the effective area from a uniform distribution
# where the smoothing lengths have been computed properly.
# This program is not written flexibly, and will only do the
# plots for one specific particle of this specific test case.
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt


import astro_meshless_surfaces as ml
from my_utils import setplotparams_multiple_plots

setplotparams_multiple_plots(for_presentation=True)


# ---------------------------
# initialize variables
# ---------------------------


# temp during rewriting
srcfile = "./snapshot_0000.hdf5"  # swift output file
ptype = "PartType0"  # for which particle type to look for
pcoord = np.array([0.5, 0.5])  # coordinates of particle to work for


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

    x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)

    # convert H to h
    #  H = h
    H = ml.get_H(h)
    pind = ml.find_index(x, y, pcoord)
    tree, nbors = ml.find_neighbours(pind, x, y, H)

    print("Computing effective surfaces")

    A_ij_Hopkins = ml.Aij_Hopkins(pind, x, y, H, m, rho, tree=tree)
    A_ij_Ivanova = ml.Aij_Ivanova(pind, x, y, H, tree=tree)

    x_ij = ml.x_ij(pind, x, y, H, nbors=nbors)

    print("Sum Hopkins:", np.sum(A_ij_Hopkins, axis=0))
    print("Sum Ivanova:", np.sum(A_ij_Ivanova, axis=0))

    Hnorm = np.sum(np.sqrt(A_ij_Hopkins[:, 0] ** 2 + A_ij_Hopkins[:, 1] ** 2))
    Inorm = np.sum(np.sqrt(A_ij_Ivanova[:, 0] ** 2 + A_ij_Ivanova[:, 1] ** 2))
    print("")
    print("Sum norm Hopkins:", Hnorm)
    print("Sum norm Ivanova:", Inorm)

    print("")

    print(r" Ratios Hopkins/Ivanova $|A_{ij}|$     & particle position \\")
    print("\hline")
    dist = np.zeros(len(nbors), dtype=np.float)
    for i, n in enumerate(nbors):
        dx, dy = ml.get_dx(x[pind], x[n], y[pind], y[n])
        dist[i] = np.sqrt(dx ** 2 + dy ** 2)

    inds = np.argsort(dist)

    for ind in range(len(nbors)):
        i = inds[ind]
        AI = np.sqrt(A_ij_Ivanova[i][0] ** 2 + A_ij_Ivanova[i][1] ** 2)
        AH = np.sqrt(A_ij_Hopkins[i][0] ** 2 + A_ij_Hopkins[i][1] ** 2)
        print(
            r"{0:8.6f}    & ({1:6.3f}, {2:6.3f} )\\".format(
                AH / AI, x[nbors[i]], y[nbors[i]]
            )
        )

    print("Plotting")

    # plot particles in order of distance:
    # closer ones first, so that you still can see the short arrows

    dist = np.zeros(len(nbors))
    for i, n in enumerate(nbors):
        dist[i] = (x[n] - pcoord[0]) ** 2 + (y[n] - pcoord[1]) ** 2

    args = np.argsort(dist)

    fig = plt.figure(figsize=(11, 5.5))
    #  fig = plt.figure(figsize=(34, 9))
    ax1 = fig.add_subplot(121, aspect="equal")
    ax2 = fig.add_subplot(122, aspect="equal")

    pointsize = 100
    arrwidth = 2

    for ax in [ax1, ax2]:  # , ax3, ax4]:
        ax.set_facecolor("lavender")
        ax.scatter(x[pind], y[pind], c="k", s=pointsize * 2)
        ax.set_xlim((0.25, 0.75))
        ax.set_ylim((0.25, 0.75))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        for i in range(len(nbors)):

            ii = args[i]
            n = nbors[ii]

            cc = i
            while cc > ncolrs - 1:
                cc -= ncolrs
            col = fullcolorlist[cc]

            arrwidth = 2

            ax.scatter(x[n], y[n], c=col, s=pointsize, zorder=0, lw=1, edgecolor="k")

    for i in range(len(nbors)):
        ii = args[i]
        n = nbors[ii]

        cc = i
        while cc > ncolrs - 1:
            cc -= ncolrs
        col = fullcolorlist[cc]

        ax1.arrow(
            x_ij[ii][0],
            x_ij[ii][1],
            A_ij_Hopkins[ii][0],
            A_ij_Hopkins[ii][1],
            color="k",
            lw=arrwidth + 1,
            zorder=9 + i,
        )
        ax1.arrow(
            x_ij[ii][0],
            x_ij[ii][1],
            A_ij_Hopkins[ii][0],
            A_ij_Hopkins[ii][1],
            color=col,
            lw=arrwidth,
            zorder=10 + i,
        )

        ax2.arrow(
            x_ij[ii][0],
            x_ij[ii][1],
            A_ij_Ivanova[ii][0],
            A_ij_Ivanova[ii][1],
            color="k",
            lw=arrwidth + 1,
            zorder=9 + i,
        )
        ax2.arrow(
            x_ij[ii][0],
            x_ij[ii][1],
            A_ij_Ivanova[ii][0],
            A_ij_Ivanova[ii][1],
            color=col,
            lw=arrwidth,
            zorder=10 + i,
        )

    ax1.set_title(
        r"Hopkins $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$"
    )  # , fontsize=18, pad=12)

    ax2.set_title(
        r"Ivanova $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$"
    )  # , fontsize=18, pad=12)

    plt.savefig("effective_area_hopkins_vs_ivanova.png")


if __name__ == "__main__":
    main()
