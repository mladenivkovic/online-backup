#!/usr/bin/env python3

# --------------------------------------------------------
# Plot the effective area as used for the meshless
# methods by Hopkins 2015
# Usage: just run the script
# some parameters can be set manually below
# --------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


# -------------------------
# Vars for manual setup
# -------------------------


nx = (
    100
)  # number of grid points in any direction for plots. For 3D, 20 barely works for me.
# also note that this is for 3D, so don't overdo it?
boxlen = 1  # box size in every direction
dx = boxlen / nx

nslices = (
    5  # if plotting slices of the values along a direction, set how many slices to do
)

# particle coordinates
part1 = [0, 0, 0]
part2 = [1, 1, 1]

# kernel length
h = 2

# associated particle volume V_i = integral over all space psi(x) dV
V = 0.5
# integral of cubic spline kernel is 3/(4*pi) for both particles, so
# normalisation gives 1/(2 * 3/4pi) * 3/4pi = 1/2


def W(q, h):
    """
    cubic spline kernel
    """
    sigma = 10.0 / (7 * np.pi * h ** 2)
    if q < 1:
        return 0.25 * (2 - q) ** 3 - (1 - q) ** 3
    elif q < 2:
        return 0.25 * (2 - q) ** 3
    else:
        return 0


def psi(x, part):
    """
    UNNORMALIZED Volume fraction at position x of some particle part
    """
    q = (
        np.sqrt((x[0] - part[0]) ** 2 + (x[1] - part[1]) ** 2 + (x[2] - part[2]) ** 2)
        / h
    )

    return W(q, h)


def get_A():
    """
    Get a 3d array of the surface function A for every
    of nx positions in the interval (0, boxlen)
    First computes the gradients of psi_i and psi_j
    using simple finite difference method

    returns Ax, Ay, Az: The value of every component at
            every point in space to be plotted
    """

    psi_i = np.zeros((nx + 1, nx + 1, nx + 1), dtype=np.float)
    psi_j = np.zeros((nx + 1, nx + 1, nx + 1), dtype=np.float)

    # compute psi_i, psi_j
    print("computing psi")
    for i in range(nx + 1):
        for j in range(nx + 1):
            for k in range(nx + 1):
                pos = [(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx]
                p1 = psi(pos, part1)
                p2 = psi(pos, part2)
                psisum = p1 + p2
                psi_i[i, j, k] = p1 / psisum
                psi_j[i, j, k] = p2 / psisum

    print("computing psi gradients")
    # now estimate gradients cheaply
    dpsi_ix = (psi_i[1:, :-1, :-1] - psi_i[:-1, :-1, :-1]) / dx
    dpsi_iy = (psi_i[:-1, 1:, :-1] - psi_i[:-1, :-1, :-1]) / dx
    dpsi_iz = (psi_i[:-1, :-1, 1:] - psi_i[:-1, :-1, :-1]) / dx

    dpsi_jx = (psi_j[1:, :-1, :-1] - psi_j[:-1, :-1, :-1]) / dx
    dpsi_jy = (psi_j[:-1, 1:, :-1] - psi_j[:-1, :-1, :-1]) / dx
    dpsi_jz = (psi_j[:-1, :-1, 1:] - psi_j[:-1, :-1, :-1]) / dx

    print("computing A")
    Ax = V * dpsi_jx - V * dpsi_ix
    Ay = V * dpsi_jy - V * dpsi_iy
    Az = V * dpsi_jz - V * dpsi_iz

    return Ax, Ay, Az


# =================================
def plot_3d(Ax, Ay, Az):
    # =================================
    """
    Do a 3d scatterplot of |A|
    Don't really recommend it, it's reaaaally slow with matplotlib
    """

    A = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)

    # -----------------------
    # Plot results
    # -----------------------

    print("plotting A in 3d")
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")

    cmin = A.min()
    cmax = A.max()

    for i in range(nx):
        x = (i + 0.5) * dx
        for j in range(nx):
            y = (j + 0.5) * dx
            for k in range(nx):
                z = (k + 0.5) * dx

                sp = ax1.scatter3D(
                    x, y, z, c=[A[i, j, k]], cmap="YlGnBu_r", vmin=cmin, vmax=cmax
                )

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    plt.colorbar(sp)

    plt.show()


# =========================================
def plot_2d_slices(Ax, Ay, Az):
    # =========================================
    """
    Plots nslices slices along each axis, in every direction.
    Meaning:
        For the gradient in x-direction, plot the values
        along x, y, z at different positions
    """

    print("plotting 2d slices")

    for A, direction in [(Ax, "x"), (Ay, "y"), (Az, "z")]:

        cmin = A.min()
        cmax = A.max()

        fig = plt.figure(figsize=(20, 12))

        axes = []
        for r in range(3):
            newax = []
            for c in range(nslices):
                newax.append(fig.add_subplot(3, nslices, nslices * r + c + 1))

            axes.append(newax)

        for s in range(nslices):
            ind = int(nx / nslices * (s + 0.5))
            sliceval = boxlen / nslices * (s + 0.5)

            ax1 = axes[0][s]
            ax2 = axes[1][s]
            ax3 = axes[2][s]

            i1 = ax1.imshow(
                A[ind, :, :],
                cmap="YlGnBu_r",
                origin="lower",
                extent=(0, boxlen, 0, boxlen),
                vmin=cmin,
                vmax=cmax,
            )
            ax1.set_xlabel("y")
            ax1.set_ylabel("z")
            ax1.set_title("along x = {0:6.3f}".format(sliceval))
            fig.colorbar(i1, ax=ax1)

            i2 = ax2.imshow(
                A[:, ind, :],
                cmap="YlGnBu_r",
                origin="lower",
                extent=(0, boxlen, 0, boxlen),
                vmin=cmin,
                vmax=cmax,
            )
            ax2.set_xlabel("x")
            ax2.set_ylabel("z")
            ax2.set_title("along y = {0:6.3f}".format(sliceval))
            fig.colorbar(i2, ax=ax2)

            i3 = ax3.imshow(
                A[:, :, ind],
                cmap="YlGnBu_r",
                origin="lower",
                extent=(0, boxlen, 0, boxlen),
                vmin=cmin,
                vmax=cmax,
            )
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_title("along z = {0:6.3f}".format(sliceval))
            fig.colorbar(i3, ax=ax3)

        fig.suptitle("Slices for " + direction + " component of A")

        #  plt.show()
        fname = "meshless_surface_slices_" + direction + ".png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        print("saved ", fname)
        plt.close()


# =========================================
def plot_2d_sums(Ax, Ay, Az):
    # =========================================
    """
    Sums up values of every component of A along all 3 axes
    and plots them
    Meaning:
        For the gradient in x-direction, plot the
        cumulative valuesi along x, y, z directions

    In the and, also plot |A| in different directions
    """

    print("plotting 2d cumulative")

    Aabs = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)

    for A, direction in [(Ax, "x"), (Ay, "y"), (Az, "z"), (Aabs, "norm")]:

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        i1 = ax1.imshow(
            np.sum(A, 0), cmap="YlGnBu_r", origin="lower", extent=(0, boxlen, 0, boxlen)
        )
        ax1.set_xlabel("y")
        ax1.set_ylabel("z")
        ax3.set_title("along x axis")
        fig.colorbar(i1, ax=ax1)

        i2 = ax2.imshow(
            np.sum(A, 1), cmap="YlGnBu_r", origin="lower", extent=(0, boxlen, 0, boxlen)
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        ax3.set_title("along y axis")
        fig.colorbar(i2, ax=ax2)

        i3 = ax3.imshow(
            np.sum(A, 2), cmap="YlGnBu_r", origin="lower", extent=(0, boxlen, 0, boxlen)
        )
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_title("along z axis")
        fig.colorbar(i3, ax=ax3)

        fig.suptitle("Cumulative values of " + direction + " component of A")

        #  plt.show()
        fname = "meshless_surface_cumulative_" + direction + ".png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        print("saved ", fname)
        plt.close()


# =================================
if __name__ == "__main__":
    # =================================

    Ax, Ay, Az = get_A()

    #  plot_3d(Ax, Ay, Az)
    plot_2d_slices(Ax, Ay, Az)
    plot_2d_sums(Ax, Ay, Az)
