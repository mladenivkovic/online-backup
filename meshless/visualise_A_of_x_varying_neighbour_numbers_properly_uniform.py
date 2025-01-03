#!/usr/bin/env python3

# ===============================================================
# Compute A(x) between two specified particles at various
# positions x
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size, ImageGrid
import astro_meshless_surfaces as ml

raise ImportError("Integrand functions in the astro_meshless_surfaces are deprecated.")


ptype = "PartType0"  # for which particle type to look for


# border limits for plots
lowlim = 0.43
uplim = 0.52
nx = 10
tol = (
    0.025  # tolerance for float comparison; easier to find particles you're looking for
)


kernels = ml.kernels
kfacts = ml.kernelfacts


def get_sample_size():
    """
    Count how many files we're dealing with
    Assumes snapshots start with "snaphsot-" string
    """

    import os
    import numpy as np

    filelist = os.listdir()
    snaplist = []
    for f in filelist:
        if f.startswith("snapshot-"):
            snaplist.append(f)

    snaplist.sort()
    eta_facts = [0 for s in snaplist]

    for i, snap in enumerate(snaplist):
        junk, dash, rest = snap.partition("snapshot-")
        f, underzero, junk = rest.partition("_0000")
        eta_facts[i] = float(f)

    return eta_facts, snaplist


def main():

    # -----------------------------
    # Part1 : compute all A
    # -----------------------------

    print("Computing effective surfaces")

    eta_facts, srcfiles = get_sample_size()

    # ----------------------
    # Set up figure
    # ----------------------

    nrows = len(eta_facts)
    ncols = len(kernels)

    Ax_list = [[None for c in range(ncols)] for r in range(nrows)]
    Ay_list = [[None for c in range(ncols)] for r in range(nrows)]
    Anorm_list = [[None for c in range(ncols)] for r in range(nrows)]

    dx = (uplim - lowlim) / nx

    #  gauss = kernels.index('gaussian')
    # GAUSS WAS REMOVED

    # --------------------------------
    # Loop over all files
    # --------------------------------

    for row, eta in enumerate(eta_facts):
        print()
        print("working for eta =", eta)
        print("================================")

        x, y, h, rho, m, ids, npart = ml.read_file(srcfiles[row], ptype)

        # find where particles i (0.45, 0.45) and j (0.5, 0.5) are
        iind = None
        jind = None

        for i in range(x.shape[0]):
            if abs(x[i] - 0.45) < tol and abs(y[i] - 0.45) < tol:
                iind = i
            if abs(x[i] - 0.5) < tol and abs(y[i] - 0.5) < tol:
                jind = i

        if iind is None or jind is None:
            raise ValueError("iind=", iind, "jind=", jind)

        # -------------------------
        # loop over all kernels
        # ------------------------

        for col, kernel in enumerate(kernels):

            # Compute gaussian only once. It doesn't depend on eta, as it has no compact support.
            # GAUSSIAN WAS REMOVED
            #  if kernel == 'gaussian':
            #      if Ax_list[0][gauss] is not None:
            #          continue

            print("working for ", kernel)

            # translate h to H
            H = ml.get_H(h, kernel)

            A = np.zeros(
                (nx, nx, 2), dtype=np.float
            )  # storing computed effective surfaces
            # ---------------------
            # compute A
            # ---------------------

            for i in range(nx):
                xx = lowlim + dx * i

                if i % 10 == 0:
                    print("i =", i, "/", nx)

                for j in range(nx):
                    yy = lowlim + dx * j

                    hh = ml.h_of_x(
                        xx, yy, x, y, H, m, rho, kernel=kernel, fact=kfacts[col]
                    )

                    A[j, i] = ml.Integrand_Aij_Ivanova(
                        iind,
                        jind,
                        xx,
                        yy,
                        hh,
                        x,
                        y,
                        H,
                        m,
                        rho,
                        kernel=kernel,
                        fact=kfacts[col],
                    )  # not a typo: need A[j,i] for imshow
                    #  for quick checks of the plotting results
                    #  A[j,i] =  row + np.random.random()/10 # not a typo: need A[j,i] for imshow

            Ax = A[:, :, 0]
            Ay = A[:, :, 1]
            Anorm = np.sqrt(Ax ** 2 + Ay ** 2)

            Ax_list[row][col] = Ax
            Ay_list[row][col] = Ay
            Anorm_list[row][col] = Anorm

    # if there is a gaussian kernel in there
    # GAUSS WAS REMOVED
    #  if gauss is not None:
    #      for r in range(1, nrows):
    #          Ax_list[r][gauss] = Ax_list[0][gauss]
    #          Ay_list[r][gauss] = Ay_list[0][gauss]
    #          Anorm_list[r][gauss] = Anorm_list[0][gauss]

    # ------------------------------------
    # Now plot it
    # ------------------------------------

    As = [Ax_list, Ay_list, Anorm_list]
    title_prefix = ["x-component", "y-component", "norm"]

    for f in range(len(title_prefix)):

        print("plotting", title_prefix[f])

        A = As[f]

        fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))

        axrows = [[] for r in range(nrows)]
        for r in range(nrows):
            axcols = [None for c in range(ncols)]

            axcols = ImageGrid(
                fig,
                (nrows, 1, r + 1),
                nrows_ncols=(1, ncols),
                axes_pad=0.0,
                share_all=True,
                label_mode="L",
                cbar_mode="edge",
                cbar_location="right",
                cbar_size="7%",
                cbar_pad="2%",
            )
            axrows[r] = axcols

        inds = np.argsort(ids)

        mask = np.logical_and(x >= lowlim - tol, x <= uplim + tol)
        mask = np.logical_and(mask, y >= lowlim - tol)
        mask = np.logical_and(mask, y <= uplim + tol)

        ec = "black"
        lw = 2
        cmap = "YlGnBu_r"

        for row in range(nrows):
            axcols = axrows[row]

            minval = min([np.min(A[row][c]) for c in range(ncols)])
            maxval = max([np.max(A[row][c]) for c in range(ncols)])

            for col, ax in enumerate(axcols):

                im = ax.imshow(
                    A[row][col],
                    origin="lower",
                    vmin=minval,
                    vmax=maxval,
                    cmap=cmap,
                    extent=(lowlim, uplim, lowlim, uplim),
                    #  norm=matplotlib.colors.SymLogNorm(1e-3),
                    zorder=1,
                )

                # superpose particles

                # plot neighbours (and the ones you drew anyway)
                ps = 50
                fc = "grey"
                ax.scatter(
                    x[mask], y[mask], s=ps, lw=lw, facecolor=fc, edgecolor=ec, zorder=2
                )

                # plot the chosen one
                ps = 100
                fc = "white"
                ax.scatter(
                    x[iind], y[iind], s=ps, lw=lw, facecolor=fc, edgecolor=ec, zorder=3
                )

                # plot central (and the ones you drew anyway)
                fc = "black"
                ax.scatter(
                    x[jind], y[jind], s=ps, lw=lw, facecolor=fc, edgecolor=ec, zorder=4
                )

                ax.set_xlim((lowlim, uplim))
                ax.set_ylim((lowlim, uplim))

                # cosmetics
                if col > 0:
                    left = False
                else:
                    left = True
                if row == len(eta_facts) - 1:
                    bottom = True
                else:
                    bottom = False

                ax.tick_params(
                    axis="both",  # changes apply to the x-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=bottom,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=left,  # ticks along the top edge are off
                    right=False,  # ticks along the top edge are off
                    labelbottom=bottom,  # labels along the bottom edge are off
                    labeltop=False,  # labels along the bottom edge are off
                    labelleft=left,  # labels along the bottom edge are off
                    labelright=False,
                )  # labels along the bottom edge are off

                if row == 0:
                    ax.set_title(kernels[col] + " kernel", fontsize=14)
                if col == 0:
                    ax.set_ylabel(r"$\eta = $ " + str(eta_facts[row]) + r"$\eta_0$")

            # Add colorbar to every row
            axcols.cbar_axes[0].colorbar(im)

        fig.suptitle(
            title_prefix[f]
            + r" of integrand of effective Area $\mathbf{A}_{ij}(\mathbf{x}) = \psi_i(\mathbf{x}) \nabla \psi_j(\mathbf{x}) - \psi_j(\mathbf{x}) \nabla \psi_i(\mathbf{x})$ of a particle (white) w.r.t. the central particle (black)"
            "\n"
            "in a uniform distribution for different kernels for $\eta_0 = 1.23485$",
            fontsize=18,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(
            "effective_area_A_of_x_uniform_different_kernels_properly_uniform-"
            + title_prefix[f]
            + ".png",
            dpi=150,
        )
        plt.close()

    print("finished.")

    return


if __name__ == "__main__":
    main()
