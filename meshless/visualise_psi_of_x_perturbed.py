#!/usr/bin/env python3


# ===============================================================
# Plot \psi(x) for a chosen particle as a function of x
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size, ImageGrid
from matplotlib import colors as clrs


import astro_meshless_surfaces as ml
import my_utils

my_utils.setplotparams_multiple_plots(for_presentation=True)


# ---------------------------
# initialize variables
# ---------------------------


# temp during rewriting
srcfile = "./snapshot_0000.hdf5"  # swift output file
ptype = "PartType0"  # for which particle type to look for
pcoord = np.array([0.5, 0.5])  # coordinates of particle to work for

nx = 200
lowlim = 0.3
uplim = 0.7


def main():

    x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)
    eta = ml.read_eta_from_file(srcfile, ptype)

    # convert H to h
    H = ml.get_H(h)
    pind = ml.find_index(x, y, pcoord)
    tree, nbors = ml.find_neighbours(pind, x, y, H)

    # compute psi_i(x)
    psi = np.zeros((nx, nx), dtype=np.float)
    dx = (uplim - lowlim) / (nx)

    for i in range(nx):
        if i % 10 == 0:
            print("i = ", i)
        for j in range(nx):
            xx = lowlim + (i + 0.5) * dx
            yy = lowlim + (j + 0.5) * dx

            tree, neighs = ml.find_neighbours_arbitrary_x(xx, yy, x, y, eta, tree=tree)
            ind = neighs == pind
            if not ind.any():
                continue
                #  raise ValueError("pind not in neighbour list")

            xj = x[neighs]
            yj = y[neighs]
            Hj = H[neighs]
            Hav = np.mean(Hj)

            psi_j = ml.compute_psi_i(xx, yy, xj, yj, Hj)
            psi[j, i] = psi_j[ind] / np.sum(psi_j)

    # set up figure

    fig = plt.figure(figsize=(10, 5.5))
    axes = ImageGrid(
        fig,
        (1, 1, 1),
        nrows_ncols=(1, 2),
        axes_pad=0.85,
        share_all=True,
        aspect="equal",
        label_mode="all",
        cbar_mode="each",
        cbar_location="right",
        cbar_size="5%",
        cbar_pad="2%",
    )

    ax1, ax2 = axes

    im1 = ax1.imshow(
        psi, origin="lower", extent=(lowlim, uplim, lowlim, uplim)
    )  # , norm=clrs.SymLogNorm(linthresh=1e-10))
    ax1.set_title(r"$\psi(\mathbf{x})$, real data")  # , fontsize=18, pad=12)
    axes.cbar_axes[0].colorbar(im1)

    xx, yy = np.meshgrid(np.linspace(lowlim, uplim, nx), np.linspace(lowlim, uplim, nx))
    conts = ax2.contourf(xx, yy, psi)
    im2 = ax2.contour(conts, colors="k")
    ax2.set_title(r"$\psi(\mathbf{x})$, contour plot")  # , fontsize=18, pad=12)
    axes.cbar_axes[1].colorbar(conts)

    for ax in [ax1, ax2]:
        ax.set_xlim((lowlim, uplim))
        ax.set_ylim((lowlim, uplim))
        #  ax.set_xlabel('x')
        #  ax.set_ylabel('y')

        # plot chosen particle and neighbours
        pointsize = 70
        ax.scatter(x[pind], y[pind], c="k", s=pointsize * 2)
        for n in nbors:
            ax.scatter(
                x[n], y[n], c="white", s=pointsize, zorder=10, lw=1, edgecolor="k"
            )

    plt.savefig("psi_of_x_perturbed.png")


if __name__ == "__main__":
    main()
