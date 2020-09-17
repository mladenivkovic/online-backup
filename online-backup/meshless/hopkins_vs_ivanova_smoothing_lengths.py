#!/usr/bin/env python3

# ===============================================================
# Compute effective surfaces at different particle positions
# for different smoothing lengths
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size, ImageGrid
import h5py


import astro_meshless_surfaces as ml
from my_utils import setplotparams_multiple_plots

setplotparams_multiple_plots(wspace=0.0, hspace=0.0)

ptype = "PartType0"  # for which particle type to look for
srcdir = "ics_and_outputs"


# border limits for plots
lowlim = 0.45
uplim = 0.55
nx = 10
tol = 1e-5  # tolerance for float comparison


def main():

    # first get how many directories there are containing
    # snapshots with different etas.
    # assumes all snapshots for different etas are in
    # separate directory within 'srcdir'

    import os

    ls = os.listdir(srcdir)
    dirs = []
    smoothing_lengths = []
    for f in ls:
        fname = os.path.join(srcdir, f)
        if os.path.isdir(fname):
            dirs.append(fname)
            smoothing_lengths.append(float(f))

    smoothing_lengths = np.array(smoothing_lengths)
    sortind = np.argsort(smoothing_lengths)
    dirs = [dirs[i] for i in sortind]
    etas = smoothing_lengths[sortind]

    nrows = 2
    ncols = len(etas)

    # assume that there are equally many files for every smoothing length
    nx, filenummax, fileskip = ml.get_sample_size(dirs[0])
    #  fileskip = 100
    #  nx = 3

    Aij_Hopkins = [np.zeros((nx, nx, 2), dtype=np.float) for e in etas]
    Aij_Ivanova = [np.zeros((nx, nx, 2), dtype=np.float) for e in etas]

    A_list = [Aij_Hopkins, Aij_Ivanova]
    A_list_funcs = [ml.Aij_Hopkins, ml.Aij_Ivanova]

    for e, eta in enumerate(etas):

        prefix = dirs[e]

        # loop over all files
        ii = 0
        jj = 0
        for i in range(1, filenummax + 1, fileskip):
            for j in range(1, filenummax + 1, fileskip):

                srcfile = os.path.join(
                    prefix,
                    "snapshot-"
                    + str(i).zfill(3)
                    + "-"
                    + str(j).zfill(3)
                    + "_0000.hdf5",
                )
                print("working for ", srcfile)

                x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)
                H = ml.get_H(h)

                cind = ml.find_central_particle(npart, ids)
                pind = ml.find_added_particle(ids)
                tree, nbors = ml.find_neighbours(pind, x, y, H)
                ind = nbors == cind
                if not ind.any():
                    print(nbors)
                    print(x[nbors])
                    print(y[nbors])
                    print("central:", cind, x[cind], y[cind])
                    print("displaced:", pind, x[pind], y[pind], 2 * h[pind])
                    jj += 1
                    continue

                for a in range(len(A_list)):
                    #  A[e][jj,ii] = np.random.uniform()
                    resI = ml.Aij_Ivanova(pind, x, y, H, tree=tree)
                    Aij_Ivanova[e][jj, ii] = resI[ind]
                    resH = ml.Aij_Hopkins(pind, x, y, H, m, rho, tree=tree)
                    Aij_Hopkins[e][jj, ii] = resH[ind]

                jj += 1

            ii += 1
            jj = 0

    Anorm_Hopkins = [np.zeros((nx, nx), dtype=np.float) for e in etas]
    Anorm_Ivanova = [np.zeros((nx, nx), dtype=np.float) for e in etas]

    for e, eta in enumerate(etas):
        Ax = Aij_Hopkins[e][:, :, 0]
        Ay = Aij_Hopkins[e][:, :, 1]
        Anorm_Hopkins[e] = np.sqrt(Ax ** 2 + Ay ** 2)

        Ax = Aij_Ivanova[e][:, :, 0]
        Ay = Aij_Ivanova[e][:, :, 1]
        Anorm_Ivanova[e] = np.sqrt(Ax ** 2 + Ay ** 2)

    fig = plt.figure(figsize=(ncols * 5, nrows * 5.5))

    inds = np.argsort(ids)

    mask = np.logical_and(x >= lowlim - tol, x <= uplim + tol)
    mask = np.logical_and(mask, y >= lowlim - tol)
    mask = np.logical_and(mask, y <= uplim + tol)
    mask[pind] = False

    dx = (uplim - lowlim) / nx

    uplim_plot = uplim  # + 0.5*dx
    lowlim_plot = lowlim  # - 0.5*dx

    cmap = "YlGnBu_r"
    lw = 2
    ec = "black"

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

            if col == 0:
                if row == 0:
                    ax.set_ylabel(r"Hopkins $|\mathbf{A}_{ij}|$", fontsize=14)
                if row == 1:
                    ax.set_ylabel(r"Ivanova $|\mathbf{A}_{ij}|$", fontsize=14)
            if row == 0:
                ax.set_title(
                    r"$\eta =$ " + "{0:5.2f}".format(etas[col]) + r" $\eta_0$",
                    fontsize=14,
                )

            # Add colorbar to every row
            axcols.cbar_axes[col].colorbar(im)

    fig.suptitle(
        r"Effective Area $\mathbf{A}_{ij}$ of a particle w.r.t. the central particle in a uniform distribution for different smoothing lengths with $\eta = \alpha \eta_0$ for $\eta_0 = 1.2348$",
        fontsize=14,
    )

    plt.savefig("different_smoothing_lengths.png")

    print("finished.")

    return


if __name__ == "__main__":
    main()
