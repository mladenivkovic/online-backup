#!/usr/bin/env python3


# ===============================================================
# Check by computing the effective surface a la Hopkins using
# two different ways and pray that it gives identical results.
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt


import astro_meshless_surfaces as ml


# ---------------------------
# initialize variables
# ---------------------------


srcfile = "./snapshot_perturbed.hdf5"  # swift output file
ptype = "PartType0"  # for which particle type to look for
pcoords = [[0.5, 0.5], [0.7, 0.7]]  # coordinates of particle to work for

verbose = True  # how talkative the code is
print_by_particle = False  # whether to print differences for each particle separately


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


def extrapolate(x, y, pind, n):
    """
    Extrapolate coordinates for particle-particle line
    """

    dx = x[pind] - x[n]
    dy = y[pind] - y[n]

    m = dy / dx

    if m == 0:
        x0 = 0
        y0 = y[pind]
        x1 = 1
        y1 = y[pind]
        return [x0, x1], [y0, y1]

    if dx < 0:
        xn = 1
        yn = y[pind] + m * (xn - x[pind])
        return [x[pind], xn], [y[pind], yn]
    else:
        xn = 0
        yn = y[pind] + m * (xn - x[pind])
        return [x[pind], xn], [y[pind], yn]


def main():

    x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)

    # convert H to h
    H = ml.get_H(h)

    # prepare figure
    nrows = len(pcoords)
    ncols = 2
    fig = plt.figure(figsize=(5 * ncols + 0.5, 5 * nrows + 0.5))

    count = 0
    for row, pcoord in enumerate(pcoords):

        print("Working for particle at", pcoord)

        pind = ml.find_index(x, y, pcoord, tolerance=0.05)
        nbors = ml.find_neighbours(pind, x, y, H)

        print("Computing effective surfaces")

        A_ij_Hopkins = ml.Aij_Hopkins(pind, x, y, H, m, rho)
        A_ij_Hopkins_v2 = ml.Aij_Hopkins_v2(pind, x, y, H, m, rho)

        x_ij = ml.x_ij(pind, x, y, H, nbors=nbors)

        print("Plotting")

        ax1 = fig.add_subplot(nrows, ncols, count + 1, aspect="equal")
        ax2 = fig.add_subplot(nrows, ncols, count + 2, aspect="equal")
        count += ncols

        pointsize = 100
        xmin = pcoord[0] - 0.25
        xmax = pcoord[0] + 0.25
        ymin = pcoord[1] - 0.25
        ymax = pcoord[1] + 0.25

        # plot particles in order of distance:
        # closer ones first, so that you still can see the short arrows

        dist = np.zeros(len(nbors))
        for i, n in enumerate(nbors):
            dist[i] = (x[n] - pcoord[0]) ** 2 + (y[n] - pcoord[1]) ** 2

        args = np.argsort(dist)

        if verbose:
            print("Sum Hopkins:   ", np.sum(A_ij_Hopkins, axis=0))
            print("Sum Hopkins_v2:", np.sum(A_ij_Hopkins_v2, axis=0))

            abs1 = np.sqrt(A_ij_Hopkins[:, 0] ** 2 + A_ij_Hopkins[:, 1] ** 2)
            abs2 = np.sqrt(A_ij_Hopkins_v2[:, 0] ** 2 + A_ij_Hopkins_v2[:, 1] ** 2)

            print("Sum abs Hopkins:   ", np.sum(abs1))
            print("Sum abs Hopkins_v2:", np.sum(abs2))

            #  print("Ratio Hopkins/Ivanova", np.sum(abs1)/np.sum(abs_i))
            V_i = ml.V(pind, m, rho)
            #  print(V_i)
            R = np.sqrt(V_i / np.pi)
            print("spheric surface", 2 * np.pi * R)
            #  print("mean h^2", np.mean(h**2))

            print()
            print("===================================================")
            print("===================================================")
            print("===================================================")

        for ax in [ax1, ax2]:
            ax.set_facecolor("lavender")
            ax.scatter(x[pind], y[pind], c="k", s=pointsize * 2)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
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

                if print_by_particle:
                    print("Particle colour", col)
                    print(
                        "Max difference x:",
                        (A_ij_Hopkins[ii, 0] - A_ij_Hopkins_v2[ii, 0])
                        / A_ij_Hopkins[ii, 0],
                    )
                    print(
                        "Max difference y:",
                        (A_ij_Hopkins[ii, 1] - A_ij_Hopkins_v2[ii, 1])
                        / A_ij_Hopkins[ii, 1],
                    )
                    print(
                        "x v1:",
                        A_ij_Hopkins[ii, 0],
                        "v2:",
                        A_ij_Hopkins_v2[ii, 0],
                        A_ij_Hopkins[ii, 0] - A_ij_Hopkins_v2[ii, 0],
                    )
                    print(
                        "y v1:",
                        A_ij_Hopkins[ii, 1],
                        "v2:",
                        A_ij_Hopkins_v2[ii, 1],
                        A_ij_Hopkins[ii, 1] - A_ij_Hopkins_v2[ii, 1],
                    )
                    abs1 = np.sqrt(A_ij_Hopkins[ii, 0] ** 2 + A_ij_Hopkins[ii, 1] ** 2)
                    abs2 = np.sqrt(
                        A_ij_Hopkins_v2[ii, 0] ** 2 + A_ij_Hopkins_v2[ii, 1] ** 2
                    )
                    print("Max difference norm:", (abs1 - abs2) / abs1)
                    print()

                # straight line
                xx, yy = extrapolate(x, y, pind, n)
                ax.plot(xx, yy, c=col, zorder=0, lw=1)
                # plot points
                ax.scatter(
                    x[n], y[n], c=col, s=pointsize, zorder=1, lw=1, edgecolor="k"
                )

        for i in range(len(nbors)):

            ii = args[i]
            n = nbors[ii]

            cc = i
            while cc > ncolrs - 1:
                cc -= ncolrs
            col = fullcolorlist[cc]

            arrwidth = 2

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
                A_ij_Hopkins_v2[ii][0],
                A_ij_Hopkins_v2[ii][1],
                color=col,
                lw=arrwidth,
                zorder=10 + i,
            )

        ax1.set_title(
            r"Hopkins $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$",
            fontsize=12,
            pad=12,
        )

        ax2.set_title(
            r"Hopkins_v2 $\mathbf{A}_{ij}$ at $\mathbf{x}_{ij} = \mathbf{x}_i + \frac{h_i}{h_i+h_j}(\mathbf{x}_j - \mathbf{x}_i)$",
            fontsize=12,
            pad=12,
        )

    plt.tight_layout()
    plt.savefig("check_versions.png", dpi=200)


if __name__ == "__main__":
    main()
