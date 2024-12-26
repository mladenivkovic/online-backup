#!/usr/bin/env python3


# ===============================================================
# Check multiple versions for xij
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt

import astro_meshless_surfaces as ml


# ---------------------------
# initialize variables
# ---------------------------


# temp during rewriting
srcfile = "./snapshot_perturbed.hdf5"  # swift output file
ptype = "PartType0"  # for which particle type to look for
pcoords = [
    np.array([0.5, 0.5]),
    np.array([0.7, 0.7]),
]  # coordinates of particle to work for

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

arrwidth = 2

periodic = True
kernel = "cubic_spline"
L = np.array([1.0, 1.0])


def compute_xij(i, x, y, H, omega, nbors):

    # attempt 1

    #  V = np.sum(omega)
    #  Vi = 1 / omega[i]
    #  xi = x[i]
    #  yi = x[i]
    #  xij = [np.array([xi, yi]) for j in nbors]
    #
    #  for j, n in enumerate(nbors):
    #      Vj = 1 / omega[n]
    #      xj = x[n]
    #      yj = y[n]
    #      xij[j][0] += (xi * Vi - xj * Vj) / Vi
    #      xij[j][1] += (yi * Vi - yj * Vj) / Vi
    #
    #      print(xij[j])

    # attempt 2

    xij = np.zeros((len(nbors), 2), dtype=np.float)

    Vi = 1 / omega[i]
    xi = x[i]
    yi = x[i]

    for j, n in enumerate(nbors):
        Vj = 1 / omega[n]
        xj = x[n]
        yj = y[n]

        xij[j][0] = (Vi * xi + Vj * xj) / (Vi + Vj)
        xij[j][1] = (Vi * yi + Vj * yj) / (Vi + Vj)
        print(xij[j][0], xij[j][1])

    #  attempt 3

    #  xij = np.zeros((len(nbors), 2), dtype=np.float)
    #
    #  Vi = 1/omega[i]
    #  xi = x[i]
    #  yi = x[i]
    #
    #
    #  psi_i = ml.W(0., H[i])
    #
    #  for j,n in enumerate(nbors):
    #      psi_j = ml.W(0., H[n])
    #      xj = x[n]
    #      yj = y[n]
    #
    #      xij[j][0] = (psi_i * xi + psi_j * xj )/ (psi_i + psi_j)
    #      xij[j][1] = (psi_i * yi + psi_j * yj) / (psi_i + psi_j)

    return xij


# ========================
def main():
    # ========================

    x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)

    # convert H to h
    H = ml.get_H(h)

    # prepare figure
    nrows = len(pcoords)
    fig = plt.figure(figsize=(2 * 5 + 0.5, 2 * 5 + 1))
    tree = ml.get_tree(x, y, L=L, periodic=periodic)

    # compute full ivanova only once
    Aij_Ivanova_full, nbors_all, nneigh = ml.Aij_Ivanova_all(x, y, H, tree=tree)

    # compute omega
    omega = np.zeros(npart, dtype=np.float)
    for j in range(npart):
        for i, ind_n in enumerate(nbors_all[j]):
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            omega[j] += ml.psi(
                x[j],
                y[j],
                x[ind_n],
                y[ind_n],
                H[j],
                kernel=kernel,
                L=L,
                periodic=periodic,
            )
        # add self-contribution
        omega[j] += ml.psi(
            0.0, 0.0, 0.0, 0.0, H[j], kernel=kernel, L=L, periodic=periodic
        )

    count = 0
    for row, pcoord in enumerate(pcoords):

        print("Working for particle at", pcoord)

        pind = ml.find_index(x, y, pcoord)
        nbors = nbors_all[pind, : nneigh[pind]]

        Aij_Ivanova = Aij_Ivanova_full[pind][: len(nbors)]

        print(pind)
        print(x)
        print(y)
        print(H)
        print(nbors)
        x_ij_1 = ml.x_ij(pind, x, y, H, nbors=nbors)
        x_ij_2 = compute_xij(pind, x, y, H, omega, nbors)

        print("Plotting")

        ax1 = fig.add_subplot(nrows, 2, count + 1)
        ax2 = fig.add_subplot(nrows, 2, count + 2)
        count += 2

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

        axes = [ax1, ax2]
        xijs = [x_ij_1, x_ij_2]
        titles = [
            r"$x_{ij} = x_i + (x_i h_i - x_j h_j)/h_i$",
            r"$x_{ij} = x_i + (x_i V_i - x_j V_j)/V$",
        ]

        for ax, xij, title in zip(axes, xijs, titles):
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

                def extrapolate():

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

                # straight line
                xx, yy = extrapolate()
                ax.plot(xx, yy, c=col, zorder=0, lw=1)
                # plot points
                ax.scatter(
                    x[n], y[n], c=col, s=pointsize, zorder=1, lw=1, edgecolor="k"
                )

                ax.scatter(xij[ii][0], xij[ii][1], marker="s", c=col)
                #  ax.arrow(  xij[ii][0], xij[ii][1], Aij_Ivanova[ii][0], Aij_Ivanova[ii][1],
                #  color=col, lw=arrwidth, zorder=10+i)

                ax.set_title(title + r" $\mathbf{A}_{ij}$", fontsize=12, pad=12)

    plt.tight_layout()
    plt.savefig("ivanova_xij.png", dpi=200)
    #  plt.show()


if __name__ == "__main__":
    main()
