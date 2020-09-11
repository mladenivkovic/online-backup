#!/usr/bin/env python3

# ===============================================================
# Compute effective surfaces at different particle positions
# for different kernels
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size, ImageGrid
import h5py

import meshless as ms
from my_utils import setplotparams_multiple_plots

setplotparams_multiple_plots(wspace=0.0, hspace=0.0)

ptype = "PartType0"  # for which particle type to look for


# border limits for plots
lowlim = 0.40
uplim = 0.60
tol = 1e-5  # tolerance for float comparison
L = 10


kernels = ms.kernels


def main():

    # -----------------------------
    # Part1 : compute all A
    # -----------------------------

    nx, filenummax, fileskip = ms.get_sample_size()
    #  nx = 10

    ncols = len(kernels)
    nrows = 2

    Aij_Hopkins = [np.zeros((nx, nx, 2), dtype=np.float) for k in kernels]
    Aij_Ivanova = [np.zeros((nx, nx, 2), dtype=np.float) for k in kernels]

    A_list = [Aij_Hopkins, Aij_Ivanova]
    A_list_funcs = [ms.Aij_Hopkins, ms.Aij_Ivanova]

    # loop over all files
    ii = 0
    jj = 0
    for i in range(1, filenummax + 1, fileskip):
        for j in range(1, filenummax + 1, fileskip):

            srcfile = (
                "snapshot-" + str(i).zfill(3) + "-" + str(j).zfill(3) + "_0000.hdf5"
            )
            print("working for ", srcfile)

            x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)
            H = ms.get_H(h)

            cind = ms.find_central_particle(L, ids)
            pind = ms.find_added_particle(ids)
            nbors = ms.find_neighbours(pind, x, y, H)
            try:
                ind = nbors.index(cind)
            except ValueError:
                print(nbors)
                print(x[nbors])
                print(y[nbors])
                print("central:", cind, x[cind], y[cind])
                print("displaced:", pind, x[pind], y[pind], 2 * h[pind])

            for k, kernel in enumerate(kernels):
                for a in range(len(A_list)):
                    f = A_list_funcs[a]
                    A = A_list[a]
                    res = f(pind, x, y, H, m, rho, kernel=kernel)
                    A[k][jj, ii] = res[ind]
                    #  A[k][jj, ii] = np.random.uniform() / (k+1) / 100

            jj += 1

        ii += 1
        jj = 0

    Anorm_Hopkins = [np.zeros((nx, nx), dtype=np.float) for k in kernels]
    Anorm_Ivanova = [np.zeros((nx, nx), dtype=np.float) for k in kernels]

    for k, kernel in enumerate(kernels):
        Ax = Aij_Hopkins[k][:, :, 0]
        Ay = Aij_Hopkins[k][:, :, 1]
        Anorm_Hopkins[k] = np.sqrt(Ax ** 2 + Ay ** 2)

        Ax = Aij_Ivanova[k][:, :, 0]
        Ay = Aij_Ivanova[k][:, :, 1]
        Anorm_Ivanova[k] = np.sqrt(Ax ** 2 + Ay ** 2)

    fig = plt.figure(figsize=(ncols * 5, nrows * 5.5))

    inds = np.argsort(ids)

    mask = np.logical_and(x >= lowlim - tol, x <= uplim + tol)
    mask = np.logical_and(mask, y >= lowlim - tol)
    mask = np.logical_and(mask, y <= uplim + tol)
    mask[pind] = False

    ec = "black"
    lw = 2
    cmap = "YlGnBu_r"

    dx = (uplim - lowlim) / nx

    uplim_plot = uplim  # + 0.5*dx
    lowlim_plot = lowlim  # - 0.5*dx

    imgdata = [Anorm_Hopkins, Anorm_Ivanova]

    axrows = [[] for r in range(nrows)]
    for row in range(nrows):
        axcols = [None for c in range(ncols)]

        axcols = ImageGrid(
            fig,
            (nrows, 1, row + 1),
            nrows_ncols=(1, ncols),
            axes_pad=0.5,
            share_all=False,
            label_mode="L",
            cbar_mode="each",
            cbar_location="right",
            cbar_size="3%",
            cbar_pad="1%",
        )
        axrows[row] = axcols

        for col, ax in enumerate(axcols):

            im = ax.imshow(
                imgdata[row][col],
                origin="lower",
                cmap=cmap,
                #  vmin=minval, vmax=maxval, cmap=cmap,
                extent=(lowlim_plot, uplim_plot, lowlim_plot, uplim_plot),
                #  norm=matplotlib.colors.SymLogNorm(1e-3),
                zorder=1,
            )

            # superpose particles

            # plot neighbours (and the ones you drew anyway)
            ps = 100
            fc = "white"
            ax.scatter(
                x[mask], y[mask], s=ps, lw=lw, facecolor=fc, edgecolor=ec, zorder=2
            )

            # plot the chosen one
            ps = 150
            fc = "black"
            ax.scatter(
                x[cind], y[cind], s=ps, lw=lw, facecolor=fc, edgecolor=ec, zorder=3
            )

            ax.set_xlim((lowlim_plot, uplim_plot))
            ax.set_ylim((lowlim_plot, uplim_plot))
            ax.set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
            ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6])

            if col == 0:
                if row == 0:
                    ax.set_ylabel(r"Hopkins $|\mathbf{A}_{ij}|$", fontsize=14)
                if row == 1:
                    ax.set_ylabel(r"Ivanova $|\mathbf{A}_{ij}|$", fontsize=14)
            if row == 0:
                name = ms.kernel_pretty_names[col]
                ax.set_title(name, fontsize=14)

            # Add colorbar
            axcols.cbar_axes[col].colorbar(im)

    fig.suptitle(
        r"Effective Area $\mathbf{A}_{ij}$ of a particle w.r.t. the central particle (black) in a uniform distribution for different kernels",
        fontsize=14,
    )
    plt.savefig("different_kernels.png", dpi=300)

    print("finished.")

    return


if __name__ == "__main__":
    main()
