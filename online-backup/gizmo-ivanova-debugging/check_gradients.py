#!/usr/bin/env python3

#========================================================
# compare results 1:1 from swift and python outputs
# intended to compare Ivanova effective surfaces
# you should have written them in hdf5 files from SWIFT
# as extra debug output
# currently hardcoded: up to 200 neighbours
#========================================================


import numpy as np
import pickle
import h5py
import os
import meshless as ms
from my_utils import yesno

#------------------------
# Filenames
#------------------------

snap = '0000'                   # which snap to use
#  hdf_prefix = 'sodShock_'        # snapshot name prefix
#  hdf_prefix = 'perturbedPlane_'  # snapshot name prefix
hdf_prefix = 'uniformPlane_'    # snapshot name prefix

srcfile = hdf_prefix+snap+'.hdf5'

swift_dump = 'dump_swift_gradient_sum_'+snap+'.pkl'
python_dump = 'dump_my_python_gradient_sum_'+snap+'.pkl'


tolerance = 1e-3
NULL = 1e-4




#==========================================
def extract_gradients_from_snapshot():
#==========================================
    """
    Reads in, sorts out and pickle dumps from swift output
    """

    if os.path.isfile(swift_dump):
        if not yesno("Dump file", swift_dump, "already exists. Shall I overwrite it?"):
            return


    #------------------
    # read in data
    #------------------

    f = h5py.File(srcfile, 'r')
    parts = f['PartType0']
    ids = parts['ParticleIDs'][:]
    pos = parts['Coordinates'][:]
    gradsum = parts['GradientSum'][:]
    nids = parts['NeighbourIDsGrads'][:]
    nneigh = parts['nneigh_grads'][:]
    grads = parts['grads'][:]

    omega = parts['omega'][:]
    vol = parts['vol'][:]

    f.close()



    #------------------
    # sort
    #------------------

    inds = np.argsort(ids)

    gradsum = gradsum[inds]
    ids = ids[inds]
    pos = pos[inds]
    nids = nids[inds]
    nneigh = nneigh[inds] + 1 # internally initialized as -1
    grads = grads[inds]
    omega = omega[inds]
    vol = vol[inds]

    for n in range(nids.shape[0]):
        nb = nneigh[n]
        ninds = np.argsort(nids[n][:nb])
        nids[n][:nb] = nids[n][:nb][ninds]
        temp = np.empty((2*nb), dtype=np.float)
        for i, nn in enumerate(ninds):

            temp[2*i:2*i+2] = grads[n,2*nn:2*nn+2]
            #  temp[2*i] = Aijs[n, 2*nn]

        grads[n][:2*nb] = temp




    #------------------
    # dump
    #------------------

    data_dump = [grads, gradsum, nids, nneigh, omega, vol, pos, ids]
    dumpfile = open(swift_dump, 'wb')
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped swift data")



    return










#========================================
def compute_gradients_my_way():
#========================================
    """
    Compute gradients using my python module, and dump results in a pickle
    """


    if os.path.isfile(python_dump):
        if not yesno("Dump file", python_dump, "already exists. Shall I overwrite it?"):
            return

    # read data from snapshot
    #  x, y, h, rho, m, ids, npart = ms.read_file(srcfile, 'PartType0')
    # sort now so you don't have to care for it later
    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, 'PartType0', sort=True)


    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)


    # set up such that you don't need arguments in functions any more
    h = H
    fact = 1
    L = 1
    periodic = True
    kernel = 'cubic_spline'



    neighbours = [[] for i in x]

    for l in range(npart):
        # find and store all neighbours;
        neighbours[l] = ms.find_neighbours(l, x, y, h, fact=fact, L=L, periodic=periodic)

    # compute all psi_k(x_l) for all l, k
    # first index: index k of psi: psi_k(x)
    # second index: index of x_l: psi(x_l)

    psi_k_at_l = np.zeros((npart, npart), dtype=np.float)
    for k in range(npart):
        #  for l in range(npart):
        for l in neighbours[k]:
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_k_at_l[k,l] = ms.psi(x[l], y[l], x[k], y[k], h[l], kernel=kernel, fact=fact, L=L, periodic=periodic)
            #  psi_k_at_l[l,k] = ms.psi(x[l], y[l], x[k], y[k], h[l], kernel=kernel, fact=fact, L=L, periodic=periodic)

        # self contribution part: k = l +> h[k] = h[l], so use h[k] here
        psi_k_at_l[k, k] = ms.psi(0, 0, 0, 0, h[k], kernel=kernel, fact=fact, L=L, periodic=periodic) 


    omega = np.zeros(npart, dtype=np.float)

    for l in range(npart):
        # compute normalisation omega for all particles
        # needs psi_k_at_l to be computed already
        omega[l] = np.sum(psi_k_at_l[neighbours[l], l]) + psi_k_at_l[l,l]
        # omega_k = sum_l W(x_k - x_l) = sum_l psi_l(x_k) as it is currently stored in memory



    grad_psi_k_at_l = np.zeros((npart, npart, 2), dtype=np.float)
    grad_W_k_at_l = np.zeros((npart, npart, 2), dtype=np.float)


    for k in range(npart):
        for l in neighbours[k]:
            # get kernel gradients

            dx, dy = ms.get_dx(x[l], x[k], y[l], y[k], L=L, periodic=periodic)

            r = np.sqrt(dx**2 + dy**2)
            if r == 0:
                ms.grad_W_k_at_l[k, l, 0] = 0
                ms.grad_W_k_at_l[k, l, 1] = 0
            else:
                grad_W_k_at_l[k, l, 0] = ms.dWdr(r/h[l], h[l], kernel) * dx / r
                grad_W_k_at_l[k, l, 1] = ms.dWdr(r/h[l], h[l], kernel) * dy / r



    sum_grad_W = np.zeros((npart, 2), dtype=np.float)

    for l in range(npart):
        sum_grad_W[l] = np.sum(grad_W_k_at_l[neighbours[l], l], axis=0)


    # first finish computing the gradients: Need W(r, h), which is currently stored as psi
    # AS DONE IN THE NORMAL IVANOVA SCRIPT
    #  for k in range(npart):
    #      for l in neighbours[k]:
    #      #  for l in range(npart):
    #          grad_psi_k_at_l[k, l, 0] = grad_W_k_at_l[k, l, 0]/omega[l] - psi_k_at_l[k, l] * sum_grad_W[l, 0]/omega[l]**2
    #          grad_psi_k_at_l[k, l, 1] = grad_W_k_at_l[k, l, 1]/omega[l] - psi_k_at_l[k, l] * sum_grad_W[l, 1]/omega[l]**2



    nneighs = np.array([len(n) for n in neighbours], dtype=np.int)
    maxlen = np.max(nneighs)
    nids = np.zeros((npart, maxlen), dtype=np.int)

    for nb in range(npart):
        nids[nb, :nneighs[nb]] = ids[neighbours[nb]]




    data_dump = [sum_grad_W, grad_W_k_at_l, nids, nneighs, omega]
    dumpfile = open(python_dump, 'wb')
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped python data")

    return










#================================
def compare_grads():
#================================
    """
    compare the gradients you got
    """

    swift_filep = open(swift_dump, 'rb')
    python_filep = open(python_dump, 'rb')

    data_swift = pickle.load(swift_filep)
    grads_s, sum_grad_s, nids_s, nneigh_s, omega_s, vol_s, pos, ids = data_swift

    data_python = pickle.load(python_filep)
    sum_grad_p, all_grads_p, nids_p, nneigh_p, omega_p = data_python

    swift_filep.close()
    python_filep.close()


    npart = ids.shape[0]







    #----------------------------------------------
    print("Checking number of neighbours")
    #----------------------------------------------
    found_difference = False
    for p in range(npart):
        py = nneigh_p[p]
        sw = nneigh_s[p]
        if py != sw:
            found_difference = True
            print("Difference: id:", ids[p], "py:", py, "sw:", sw)

    if not found_difference:
        print("Finished, all the same.")
    else:
        print("Makes no sense to continue. Exiting")
        quit()








    #----------------------------------------------
    print("Checking neighbour IDs")
    #----------------------------------------------
    found_difference = False
    for p in range(npart):
        pyarr = nids_p[p]
        swarr = nids_s[p][:nneigh_s[p]]
        for n in range(nneigh_s[p]):
            py = pyarr[n]
            sw = swarr[n]
            if py != sw:
                found_difference = True
                print("Difference: id:", ids[p], "py:", py, "sw:", sw)

    if not found_difference:
        print("Finished, all the same.")
    else:
        print("Makes no sense to continue. Exiting")
        quit()







    #------------------------------------------------------
    print("Checking individual gradient contributions")
    #------------------------------------------------------

    found_difference = False
    for p in range(npart):
        #  for n in range(1):
        for n in range(nneigh_p[p]):
            # all_grads_p has dimensions npart x npart;
            # nids_p[n]-1 is the index we're looking for
            pyx = all_grads_p[p, nids_p[p, n]-1, 0]
            pyy = all_grads_p[p, nids_p[p, n]-1, 1]
            swx = grads_s[p, 2*n]
            swy = grads_s[p, 2*n+1]

            for P, S in [(pyx, swx), (pyy, swy)]:
                if abs(P) > NULL:
                    diff = abs(1 - S/P)
                elif abs(S) > NULL:
                    diff = abs(1 - P/S)
                else:
                    continue

                if diff > tolerance:
                    print("Found difference: ID", ids[p], "neighbour", nids_p[n], "x:", pyx, swx, "y:", pyy, swy)
                    found_difference = True

    if not found_difference:
        print("Finished, all the same.")
    else:
        quit()








#      #------------------------------------------------------
    #  print("Checking gradient sums")
    #  #------------------------------------------------------
    #
    #  print(sum_grad_p.shape)
    #  print(sum_grad_s.shape)
    #
    #  found_difference = False
    #  #  for p in range(3):
    #  for p in range(npart):
    #      #  for n in range(1):
    #
    #      pyx = sum_grad_p[p,0]
    #      pyy = sum_grad_p[p,1]
    #      swx = sum_grad_s[p,0]
    #      swy = sum_grad_s[p,1]
    #
    #      for P, S in [(pyx, swx), (pyy, swy)]:
    #          if abs(P) > NULL:
    #              diff = abs(1 - S/P)
    #          elif abs(S) > NULL:
    #              diff = abs(1 - P/S)
    #          else:
    #              continue
    #
    #          if diff > tolerance:
    #              msg = "Found difference. ID {0:6d}, x: {1:12.4E} {2:12.4E}  y: {3:12.4E} {4:12.4E}, diff: {5:12.6f}".format(ids[p], pyx, swx, pyy, swy, diff)
    #              print(msg)
    #              found_difference = True
    #
    #  if not found_difference:
    #      print("Finished, all the same.")
    #  else:
    #      quit()
#







    #------------------------------------------------------
    print("Checking volumes and normalizations")
    #------------------------------------------------------

    found_difference = False
    for p in range(npart):

        pyv = 1/omega_p[p]
        pyn = omega_p[p]
        swv = vol_s[p]
        swn = omega_s[p]

        for P, S in [(pyv, swv), (pyn, swn)]:
            if abs(P) > NULL:
                diff = abs(1 - S/P)
            elif abs(S) > NULL:
                diff = abs(1 - P/S)
            else:
                continue

            if diff > tolerance:
                print("Found difference: ID", ids[p], "v:", pyv, swv, "n:", pyn, swn)

    if not found_difference:
        print("Finished, all the same.")









    return






#==========================
def main():
#==========================
    
    extract_gradients_from_snapshot()
    compute_gradients_my_way()
    compare_grads()
    return




if __name__ == '__main__':

    main()

