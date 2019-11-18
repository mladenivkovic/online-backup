#!/usr/env/python3

# compute gradients my way

import os
import pickle
import numpy as np

import meshless as ms
from my_utils import yesno, one_arg_present

from filenames import get_srcfile, get_dumpfiles
swift_dump, part_dump, python_surface_dump, python_grad_dump = get_dumpfiles()


#========================================
def compute_gradients_my_way(periodic):
#========================================
    """
    Compute gradients using my python module, and dump results in a pickle
    """

    if os.path.isfile(python_grad_dump):
        if not yesno("Dump file", python_grad_dump, "already exists. Shall I overwrite it?"):
            return

    print("Computing Gradients")

    part_filep = open(part_dump, 'rb')
    ids = pickle.load(part_filep)
    pos = pickle.load(part_filep)
    h = pickle.load(part_filep)
    part_filep.close()

    L = ms.read_boxsize()

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    npart = x.shape[0]

    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)

    # set up such that you don't need arguments in functions any more
    fact = 1
    kernel = 'cubic_spline'

    # first get neighbour data
    print("Finding Neighbours")
    neighbour_data = ms.get_neighbour_data_for_all(x, y, H, fact=fact, L=L, periodic=periodic)

    maxneigh = neighbour_data.maxneigh
    neighbours = neighbour_data.neighbours
    nneigh = neighbour_data.nneigh
    iinds = neighbour_data.iinds

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    W_j_at_i = np.zeros((npart, maxneigh), dtype=np.float)
    omega = np.zeros(npart, dtype=np.float)


    print("Computing psi_j(x_i)")

    for j in range(npart):
        #  if (j+1)%200 == 0:
        #      print(j+1, '/', npart)
        for i, ind_n in enumerate(neighbours[j]):
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            jind = ind_n
            iind = iinds[j, i]
            W_j_at_i[jind, iind] = ms.psi(x[ind_n], y[ind_n], x[j], y[j], H[ind_n], 
                                    kernel=kernel, fact=fact, L=L, periodic=periodic)
            omega[ind_n] += W_j_at_i[jind, iind]

            if j == 148 and ind_n == 123:
                print("j=", j, "i=", i, W_j_at_i[j,i])
                dx, dy = ms.get_dx(x[ind_n], x[j], y[ind_n], y[j], L=L, periodic=periodic)
                r = np.sqrt(dx**2 + dy**2)
                print("psi:", ms.psi(x[ind_n], y[ind_n], x[j], y[j], H[ind_n], L=L, periodic=periodic))
                print("W:", ms.W(r/H[ind_n], H[ind_n]) )
                print("r/H", r/H[ind_n], r, H[ind_n], dx, dy)
                print((" Particle ID {0:8d}, "+
                        "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                        "H = {4:14.7e}, dist = {5:14.7e}, dist/H = {6:14.7e}").format(
                            ids[j], pos[j,0], pos[j,1], h[j], H[j], r, r/H[j])
                            )
                print((" Particle ID {0:8d}, "+
                        "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                        "H = {4:14.7e}, dist = {5:14.7e}, dist/H = {6:14.7e}").format(
                            ids[j], x[j], y[j], h[j], H[j], r, r/H[j])
                        )
                print(("Neighbour ID {0:8d}, "+
                        "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                        "H = {4:14.7e}").format(
                            ids[ind_n], pos[ind_n,0], pos[ind_n,1], h[ind_n], H[ind_n])
                            )
                print(("Neighbour ID {0:8d}, "+
                        "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                        "H = {4:14.7e}").format(
                            ids[ind_n], x[ind_n], y[ind_n], h[ind_n], H[ind_n])
                            )
                print()

        # add self-contribution
        omega[j] += ms.psi(0.0, 0.0, 0.0, 0.0, H[j], kernel=kernel, fact=fact, L=L, periodic=periodic)



    # compute gradients now

    # gradient of psi_j at neighbour i's position
    grad_psi_j_at_i = np.zeros((npart, maxneigh*2, 2), dtype=np.float)
    # gradient of W_j at neighbour i's position
    grad_W_j_at_i = np.zeros((npart, maxneigh*2, 2), dtype=np.float)
    # gradient sum for the same h_i
    sum_grad_W = np.zeros((npart, 2), dtype=np.float)

    dwdr = np.zeros((npart, 2*maxneigh), dtype=np.float)
    r_store = np.zeros((npart, 2*maxneigh), dtype=np.float)
    dx_store = np.zeros((npart, 2*maxneigh, 2), dtype=np.float)


    print("Computing radial gradients of psi_j(x_i)")

    for i in range(npart):
        #  if (i+1)%200 == 0:
        #      print(i+1, '/', npart)
        for j, jind in enumerate(neighbours[i]):
            dx, dy = ms.get_dx(x[i], x[jind], y[i], y[jind], L=L, periodic=periodic)
            r = np.sqrt(dx**2 + dy**2)

            iind = iinds[i, j]
            dw = ms.dWdr(r/H[i], H[i], kernel)

            grad_W_j_at_i[jind, iind, 0] = dw * dx / r
            grad_W_j_at_i[jind, iind, 1] = dw * dy / r

            sum_grad_W[i] += grad_W_j_at_i[jind, iind]

            # store other stuff
            dwdr[i, j] = dw
            r_store[i, j] = r
            dx_store[i, j, 0] = dx
            dx_store[i, j, 1] = dy



    # finish computing the gradients: Need W(r, h), which is currently stored as psi
    print("Computing cartesian gradients of psi_j(x_i)")

    for j in range(npart):
        #  if (j+1)%200 == 0:
        #      print(j+1, '/', npart)
        for i, ind_n in enumerate(neighbours[j]):
            grad_psi_j_at_i[j, i, 0] = grad_W_j_at_i[j, i, 0]/omega[ind_n] - W_j_at_i[j, i] * sum_grad_W[ind_n, 0]/omega[ind_n]**2
            grad_psi_j_at_i[j, i, 1] = grad_W_j_at_i[j, i, 1]/omega[ind_n] - W_j_at_i[j, i] * sum_grad_W[ind_n, 1]/omega[ind_n]**2


    nneighs = np.array([len(n) for n in neighbours], dtype=np.int)
    maxlen = np.max(nneighs)
    nids = np.zeros((npart, maxlen), dtype=np.int)

    for nb in range(npart):
        nids[nb, :nneighs[nb]] = ids[neighbours[nb]]




    dumpfile = open(python_grad_dump, 'wb')
    pickle.dump(grad_psi_j_at_i, dumpfile)
    pickle.dump(sum_grad_W, dumpfile)
    pickle.dump(grad_W_j_at_i, dumpfile)
    pickle.dump(dwdr, dumpfile)
    pickle.dump(W_j_at_i, dumpfile)
    pickle.dump(nids, dumpfile)
    pickle.dump(nneighs, dumpfile)
    pickle.dump(omega, dumpfile)
    pickle.dump(r_store, dumpfile)
    pickle.dump(dx_store, dumpfile)
    pickle.dump(iinds, dumpfile)
    dumpfile.close()
    print("Dumped python data")

    return






