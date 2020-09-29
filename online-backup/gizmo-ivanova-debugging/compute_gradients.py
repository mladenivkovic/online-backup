#!/usr/bin/env python3

# compute gradients my way

import os
import pickle
import numpy as np
from copy import deepcopy
from numba import jit, prange

import astro_meshless_surfaces as ml
from my_utils import yesno, one_arg_present

# local file
from filenames import get_srcfile, get_dumpfiles

swift_dump, part_dump, python_surface_dump, python_grad_dump = get_dumpfiles()


@jit(nopython=True)
def get_all_gradient_parts(
    x,
    y,
    H,
    W_j_at_i,
    omega,
    neighbours,
    nneigh,
    L=np.ones(2),
    periodic=True,
    kernel="cubic_spline",
):
    """
    Compute the actual gradients now, and store all intermediate
    computations for comparisons
    """

    npart = x.shape[0]
    maxneigh = neighbours.shape[1]

    # gradient of psi_j at neighbour i's position
    grad_psi_j_at_i = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))

    # gradient of W_j at neighbour i's position
    grad_W_j_at_i = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))

    sum_grad_W_contrib = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))

    # gradient sum for the same h_i
    sum_grad_W = np.zeros(npart * 2).reshape((npart, 2))

    dwdr = np.zeros(npart * maxneigh).reshape((npart, maxneigh))
    r_store = np.zeros(npart * maxneigh).reshape((npart, maxneigh))
    dx_store = np.zeros(npart * maxneigh * 2).reshape((npart, maxneigh, 2))

    #  print("Computing radial gradients of psi_j(x_i)")

    for j in prange(npart):
        #  if (i+1)%200 == 0:
        #      print(i+1, '/', npart)
        for i, ind_n in enumerate(neighbours[j, : nneigh[j]]):
            dx, dy = ml.get_dx(x[j], x[ind_n], y[j], y[ind_n], L=L, periodic=periodic)
            r = np.sqrt(dx ** 2 + dy ** 2)

            # now compute the term needed for the gradient sum need to do it separately: if i is neighbour of j,
            # but j is not neighbour of i, then j's contribution will be missed
            dwj = ml.dWdr(r / H[j], H[j], kernel)
            sum_grad_W[j, 0] += dwj * dx / r
            sum_grad_W[j, 1] += dwj * dy / r
            sum_grad_W_contrib[j, i, 0] = dwj * dx / r
            sum_grad_W_contrib[j, i, 1] = dwj * dy / r

            grad_W_j_at_i[j, i, 0] = dwj * dx / r
            grad_W_j_at_i[j, i, 1] = dwj * dy / r

            # store other stuff
            # store to mimick how swift writes output
            dwdr[j, i] = dwj
            r_store[j, i] = r  # symmetrical, no reason to store differently
            dx_store[j, i, 0] = dx
            dx_store[j, i, 1] = dy

    # finish computing the gradients: Need W(r, h), which is currently stored as psi
    #  print("Computing cartesian gradients of psi_j(x_i)")

    for j in prange(npart):
        #  if (j+1)%200 == 0:
        #      print(j+1, '/', npart)
        for i, ind_n in enumerate(neighbours[j, : nneigh[j]]):
            # store them so that at index j, only hj is ever used
            grad_psi_j_at_i[j, i, :] = (
                grad_W_j_at_i[j, i, :] / omega[j]
                - W_j_at_i[j, i] * sum_grad_W[j, :] / omega[j] ** 2
            )

    return (
        grad_psi_j_at_i,
        grad_W_j_at_i,
        sum_grad_W_contrib,
        sum_grad_W,
        dwdr,
        r_store,
        dx_store,
    )


def compute_gradients_my_way(periodic):
    """
    Compute gradients using my python module, and dump results in a pickle
    """

    if os.path.isfile(python_grad_dump):
        if not yesno(
            "Dump file", python_grad_dump, "already exists. Shall I overwrite it?"
        ):
            return

    print("Computing Gradients")

    part_filep = open(part_dump, "rb")
    ids = pickle.load(part_filep)
    pos = pickle.load(part_filep)
    h = pickle.load(part_filep)
    part_filep.close()

    L = ml.read_boxsize()

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    npart = x.shape[0]

    # get kernel support radius instead of smoothing length
    H = ml.get_H(h)

    # set up such that you don't need arguments in functions any more
    fact = 1
    kernel = "cubic_spline"

    # first get neighbour data
    print("Finding Neighbours")
    tree, neighbours, nneigh = ml.get_neighbours_for_all(
        x, y, H, L=L, periodic=periodic
    )

    maxneigh = neighbours.shape[1]

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    print("Computing psi_j(x_i)")
    W_j_at_i, omega = ml.get_W_j_at_i(
        x, y, H, neighbours, nneigh, L=L, periodic=periodic
    )
    W_j_at_i_output = deepcopy(W_j_at_i)

    (
        grad_psi_j_at_i,
        grad_W_j_at_i,
        sum_grad_W_contrib,
        sum_grad_W,
        dwdr,
        r_store,
        dx_store,
    ) = get_all_gradient_parts(
        x,
        y,
        H,
        W_j_at_i,
        omega,
        neighbours,
        nneigh,
        L=L,
        periodic=periodic,
        kernel=kernel,
    )

    nids = np.zeros((npart, maxneigh), dtype=np.int)

    for nb in range(npart):
        nids[nb, : nneigh[nb]] = ids[neighbours[nb, : nneigh[nb]]]

    dumpfile = open(python_grad_dump, "wb")
    pickle.dump(grad_psi_j_at_i, dumpfile)
    pickle.dump(sum_grad_W, dumpfile)
    pickle.dump(sum_grad_W_contrib, dumpfile)
    pickle.dump(dwdr, dumpfile)
    pickle.dump(W_j_at_i_output, dumpfile)
    pickle.dump(nids, dumpfile)
    pickle.dump(nneigh, dumpfile)
    pickle.dump(omega, dumpfile)
    pickle.dump(r_store, dumpfile)
    pickle.dump(dx_store, dumpfile)
    #  pickle.dump(iinds, dumpfile)
    dumpfile.close()
    print("Dumped python data")

    return
