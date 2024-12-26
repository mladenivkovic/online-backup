#!/usr/bin/env python3

# ===============================================================
# Compute A(x) between two specified particles at various
# positions x
# here also compute it for different number of neighbours considered
# ===============================================================


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import astro_meshless_surfaces as ml

raise ImportError("Integrand functions in the astro_meshless_surfaces are deprecated.")

import h5py


srcfile = "snapshot_0000.hdf5"
ptype = "PartType0"  # for which particle type to look for

boxSize = 1
L = 10

# border limits for plots
lowlimx = 0.35
uplimx = 0.55
lowlimy = 0.35
uplimy = 0.55
nx = 100
tol = 1e-5  # tolerance for float comparison

kernels = ml.kernels


def main():

    # -----------------------------
    # Part1 : compute all A
    # -----------------------------

    print("Computing effective surfaces")

    x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)

    H = ml.get_H(h)

    # chosen particle coordinates for diagonal/vertical case
    x1_case = [0.4, 0.5]  # x coords for both cases for particle 1
    y1_case = [0.4, 0.4]  # y coords for both cases for particle 1
    x2_case = [0.5, 0.5]
    y2_case = [0.5, 0.5]

    for k, kernel in enumerate(kernels):
        print("working for", kernel)

        for case, direction in enumerate(["diagonal", "vertical"]):

            print("working for", direction)

            iind = None
            jind = None

            for i in range(x.shape[0]):
                if abs(x[i] - x1_case[case]) < tol and abs(y[i] - y1_case[case]) < tol:
                    iind = i
                if abs(x[i] - x2_case[case]) < tol and abs(y[i] - y2_case[case]) < tol:
                    jind = i

            if iind is None or jind is None:
                raise ValueError("iind=", iind, "jind=", jind)

            # find hmin such that diagonal particles are still neighbours
            dxsq = (x[iind] - x[jind]) ** 2 + (y[iind] - y[jind]) ** 2
            hsq = h[iind] ** 2
            hfactmin = np.sqrt(dxsq / hsq)
            hfact = [2, 1.8, 1.6, 1.4, 1.2]
            print("hfactmin is:", hfactmin)
            print("hfacts are:", hfact)

            # update lowlims and uplims such that you don't take any points
            # where the chosen particles aren't neighbours
            global lowlimx, uplimx, lowlimy, uplimy
            if direction == "diagonal":
                lowlim = min(x1_case[case], x2_case[case], y1_case[case], y2_case[case])
                uplim = max(x1_case[case], x2_case[case], y1_case[case], y2_case[case])
                lowlimx = lowlim - (hfact[-1] / hfactmin - 1) * boxSize / L / 10
                uplimx = uplim + (hfact[-1] / hfactmin - 1) * boxSize / L / 10
                lowlimy = lowlimx
                uplimy = uplimx
            elif direction == "vertical":
                de = 0.5 * hfact[-1] * h[iind]
                lowlimx = min(x1_case[case], x2_case[case]) - de
                lowlimy = 0.5 * (y1_case[case] + y2_case[case]) - de
                uplimx = max(x1_case[case], x2_case[case]) + de
                uplimy = 0.5 * (y1_case[case] + y2_case[case]) + de

            # initialize figure

            nrows = len(hfact)
            ncols = 3
            fig = plt.figure(figsize=(3.6 * ncols, 3 * nrows))

            axrows = [None for i in range(nrows)]

            for i in range(nrows):
                axcols = [None for i in range(ncols)]
                for j in range(ncols):
                    axcols[j] = fig.add_subplot(
                        nrows, ncols, i * ncols + j + 1, aspect="equal"
                    )
                axrows[i] = axcols

            # compute A and create plots in a loop
            for row, hf in enumerate(hfact):

                print("Computing hfact=", hf)

                A = np.zeros(
                    (nx, nx, 2), dtype=np.float
                )  # storing computed effective surfaces
                dx = (uplimx - lowlimx) / nx

                for i in range(nx):
                    xx = lowlimx + dx * i

                    if i % 10 == 0:
                        print("i =", i + 1, "/", nx)

                    for j in range(nx):
                        yy = lowlimy + dx * j

                        hh = ml.h_of_x(xx, yy, x, y, H, m, rho, fact=hf)

                        A[j, i] = ml.Integrand_Aij_Ivanova(
                            iind, jind, xx, yy, hh, x, y, H, m, rho, fact=hf
                        )  # not a typo: need A[j,i] for imshow

                # -----------------------------
                # Part2: Plot results
                # -----------------------------

                print("Plotting hfact=", hf)

                Ax = A[:, :, 0]
                Ay = A[:, :, 1]
                Anorm = np.sqrt(Ax ** 2 + Ay ** 2)
                xmin = Ax.min()
                xmax = Ax.max()
                ymin = Ay.min()
                ymax = Ay.max()
                normmin = Anorm.min()
                normmax = Anorm.max()

                # reset lowlim and maxlim so cells are centered around the point they represent
                dx = (uplim - lowlim) / A.shape[0]

                cmap = "YlGnBu_r"

                ax1 = axrows[row][0]
                ax2 = axrows[row][1]
                ax3 = axrows[row][2]

                im1 = ax1.imshow(
                    Ax,
                    origin="lower",
                    vmin=xmin,
                    vmax=xmax,
                    cmap=cmap,
                    extent=(lowlimx, uplimx, lowlimy, uplimy),
                )
                im2 = ax2.imshow(
                    Ay,
                    origin="lower",
                    vmin=ymin,
                    vmax=ymax,
                    cmap=cmap,
                    extent=(lowlimx, uplimx, lowlimy, uplimy),
                )
                im3 = ax3.imshow(
                    Anorm,
                    origin="lower",
                    vmin=normmin,
                    vmax=normmax,
                    cmap=cmap,
                    extent=(lowlimx, uplimx, lowlimy, uplimy),
                )

                for ax, im in [(ax1, im1), (ax2, im2), (ax3, im3)]:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="2%", pad=0.05)
                    fig.colorbar(im, cax=cax)

                # superpose particles

                inds = np.argsort(ids)

                mask = np.logical_and(x >= lowlimx - tol, x <= uplimx + tol)
                mask = np.logical_and(mask, y >= lowlimy - tol)
                mask = np.logical_and(mask, y <= uplimy + tol)

                ps = 50
                fc = "grey"
                ec = "black"
                lw = 2

                # plot neighbours (and the ones you drew anyway)
                ax1.scatter(x[mask], y[mask], s=ps, lw=lw, facecolor=fc, edgecolor=ec)
                ax2.scatter(x[mask], y[mask], s=ps, lw=lw, facecolor=fc, edgecolor=ec)
                ax3.scatter(x[mask], y[mask], s=ps, lw=lw, facecolor=fc, edgecolor=ec)

                # plot the chosen one
                ps = 100
                fc = "white"
                ax1.scatter(x[iind], y[iind], s=ps, lw=lw, facecolor=fc, edgecolor=ec)
                ax2.scatter(x[iind], y[iind], s=ps, lw=lw, facecolor=fc, edgecolor=ec)
                ax3.scatter(x[iind], y[iind], s=ps, lw=lw, facecolor=fc, edgecolor=ec)

                # plot central (and the ones you drew anyway)
                fc = "black"
                ax1.scatter(x[jind], y[jind], s=ps, lw=lw, facecolor=fc, edgecolor=ec)
                ax2.scatter(x[jind], y[jind], s=ps, lw=lw, facecolor=fc, edgecolor=ec)
                ax3.scatter(x[jind], y[jind], s=ps, lw=lw, facecolor=fc, edgecolor=ec)

                ax1.set_xlim((lowlimx, uplimx))
                ax1.set_ylim((lowlimy, uplimy))
                ax2.set_xlim((lowlimx, uplimx))
                ax2.set_ylim((lowlimy, uplimy))
                ax3.set_xlim((lowlimx, uplimx))
                ax3.set_ylim((lowlimy, uplimy))

                ax1.set_title(r"$x$ component of $\mathbf{A}_{ij}$")
                ax2.set_title(r"$y$ component of $\mathbf{A}_{ij}$")
                ax3.set_title(r"$|\mathbf{A}_{ij}|$")
                ax1.set_xlabel("x")
                ax1.set_ylabel(
                    "neighbour cutoff =" + "{0:3.1f}".format(hf) + "h", size=12
                )
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax3.set_xlabel("x")
                ax3.set_ylabel("y")

            fig.suptitle(
                r"Integrand of Effective Area $\mathbf{A}_{ij}(\mathbf{x}) = \psi_i(\mathbf{x}) \nabla \psi_j(\mathbf{x}) - \psi_j(\mathbf{x}) \nabla \psi_i(\mathbf{x})$ of a particle (white) "
                "\n"
                r" w.r.t. the central particle (black) in a uniform distribution for "
                + kernel
                + " kernel"
            )
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            plt.savefig(
                "effective_area_A_of_x_varying_number_of_neighbours-"
                + kernel
                + "-"
                + direction
                + ".png",
                dpi=300,
            )

    print("finished.")

    return


if __name__ == "__main__":
    main()
