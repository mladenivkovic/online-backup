#!/usr/bin/env python3

#========================================================
# compare results 1:1 from swift and python outputs
# intended to compare Ivanova effective surfaces
# you should have written them in hdf5 files from SWIFT
# as extra debug output
# currently hardcoded: up to 200 neighbours
#
# You can give the snapshot number as cmd line arg, or
# hard code it.
#========================================================


import numpy as np
import pickle
import h5py
import os
import meshless as ms
from my_utils import yesno, one_arg_present

#------------------------
# Filenames
#------------------------

snap = '0000'                   # which snap to use
#  snap = '0001'                   # which snap to use
#  snap = '0002'                   # which snap to use
#  hdf_prefix = 'sodShock_'        # snapshot name prefix
#  hdf_prefix = 'sodShockSpherical_'        # snapshot name prefix
hdf_prefix = 'perturbedPlane_'  # snapshot name prefix
#  hdf_prefix = 'uniformPlane_'    # snapshot name prefix

# read in cmd line arg snapshot number if present and convert it to formatted string
snap = ms.snapstr(one_arg_present(snap))

srcfile = hdf_prefix+snap+'.hdf5'

swift_dump = 'dump_swift_gradient_sum_'+snap+'.pkl'
python_dump = 'dump_my_python_gradient_sum_'+snap+'.pkl'


#----------------------
# Behaviour params
#----------------------

tolerance = 1e-3    # relative tolerance threshold for relative float comparison: if (a - b)/a < tolerance, it's fine
NULL = 1e-6         # treat values below this as zeroes
NULL_SUMS = 5e-3    # treat sums below this as zeroes


do_break = False    # don't break after you found a difference
#  do_break = True    # don't break after you found a difference

limit_q = False      # whether to ignore differences for high q = r/H; Seems to be stupid round off errors around
q_limit = 0.97      # upper limit for q = r/H if difference is found;





#=====================
def announce():
#=====================

    print("CHECKING GRADIENTS.")
    print("tolerance is set to:", tolerance)
    print("NULL is set to:", NULL)
    print("NULL_SUMS is set to:", NULL_SUMS)
    print("do break if found difference?", do_break)
    print("use upper limit for q = r/H?", limit_q)
    if limit_q:
        print("upper limit for q = r/H is:", q_limit)
    print("---------------------------------------------------------")
    print()






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
    h = parts['SmoothingLengths'][:]
    gradsum = parts['GradientSum'][:]
    nids = parts['NeighbourIDsGrads'][:]
    nneigh = parts['nneigh_grads'][:]
    grads = parts['grads'][:]
    dwdr = parts['dwdr'][:]

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
    h = h[inds]
    nids = nids[inds]
    nneigh = nneigh[inds] + 1 # internally initialized as -1
    grads = grads[inds]
    omega = omega[inds]
    vol = vol[inds]
    dwdr = dwdr[inds]


    # sort neighbour dependent data by neighbour IDs
    for n in range(nids.shape[0]):
        nb = nneigh[n]
        # get indices of neighbours
        ninds = np.argsort(nids[n][:nb])
        # sort neighbour IDs
        nids[n][:nb] = nids[n][:nb][ninds]
        # sort dwdr
        dwdr[n][:nb] = dwdr[n][:nb][ninds]
        # sort individual gradient contributions
        temp = np.empty((2*nb), dtype=np.float)
        for i, nn in enumerate(ninds):

            temp[2*i:2*i+2] = grads[n,2*nn:2*nn+2]
            #  temp[2*i] = Aijs[n, 2*nn]

        grads[n][:2*nb] = temp




    #------------------
    # dump
    #------------------

    data_dump = [grads, gradsum, dwdr, nids, nneigh, omega, vol, pos, h, ids]
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

    for i in range(npart):
        # find and store all neighbours;
        neighbours[i] = ms.find_neighbours(i, x, y, h, fact=fact, L=L, periodic=periodic)

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    psi_j_at_i = np.zeros((npart, npart), dtype=np.float)
    for j in range(npart):
        for i in neighbours[j]:
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_j_at_i[j, i] = ms.psi(x[i], y[i], x[j], y[j], h[i], kernel=kernel, fact=fact, L=L, periodic=periodic)

        psi_j_at_i[j, j] = ms.psi(0.0, 0.0, 0.0, 0.0, h[j], kernel=kernel, fact=fact, L=L, periodic=periodic) 


    omega = np.zeros(npart, dtype=np.float)

    for i in range(npart):
        # compute normalisation omega for all particles
        omega[i] = np.sum(psi_j_at_i[neighbours[i], i]) + psi_j_at_i[i,i]
        # omega_i = sum_k W(x_k - x_i, h_k) = sum_k psi_i(x_k) as it is currently stored in memory



    grad_psi_j_at_i = np.zeros((npart, npart, 2), dtype=np.float)
    grad_W_j_at_i = np.zeros((npart, npart, 2), dtype=np.float)
    dwdr = np.zeros((npart, npart), dtype=np.float)


    for i in range(npart):
        for j in neighbours[i]:
            # get kernel gradients

            dx, dy = ms.get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)

            r = np.sqrt(dx**2 + dy**2)
            if r != 0:
                dwdr[j,i] = ms.dWdr(r/h[i], h[i], kernel)
                grad_W_j_at_i[j, i, 0] = dwdr[j,i] * dx / r
                grad_W_j_at_i[j, i, 1] = dwdr[j,i] * dy / r
            # else: zero anyway
            #  else:
            #      ms.grad_W_k_at_l[k, l, 0] = 0
            #      ms.grad_W_k_at_l[k, l, 1] = 0
            
            #  if k == 0:
            #      print("Working on neighbour", l, ids[l], dwdr[k,l],
            #          ms.dWdr(r/h[l], h[l], kernel), ms.dWdr(r/h[k], h[k], kernel))


    sum_grad_W = np.zeros((npart, 2), dtype=np.float)

    for i in range(npart):
        # you can skip the self contribution here, the gradient at r = 0 is 0
        # sum along fixed/same h_i
        sum_grad_W[i] = np.sum(grad_W_j_at_i[neighbours[i], i], axis=0)


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




    data_dump = [sum_grad_W, grad_W_j_at_i, dwdr, nids, nneighs, omega]
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
    grads_s, sum_grad_s, dwdr_s, nids_s, nneigh_s, omega_s, vol_s, pos, h, ids = data_swift
    H = ms.get_H(h)

    data_python = pickle.load(python_filep)
    sum_grad_p, all_grads_p, dwdr_p, nids_p, nneigh_p, omega_p = data_python

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
    print("Checking radial gradients")
    #------------------------------------------------------

    found_difference = False
    for p in range(npart):
        for n in range(nneigh_p[p]):
            # dwdr_p has dimensions npart x npart;
            # nids_p[n]-1 is the index we're looking for
            # remember that dwdr_p[j,i] = dw_j(x_i)/dr and dwdr_s[i, j] = dw_j(x_i)
            nind = nids_p[p,n]-1
            py = dwdr_p[nind, p]
            sw = dwdr_s[p, n]

            if abs(py) > NULL:
                diff = abs(1 - sw/py)
            elif abs(sw) > NULL:
                diff = abs(1 - py/sw)
            else:
                continue

            if diff > tolerance:

                dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1])
                r = np.sqrt(dx**2 + dy**2)

                if limit_q:
                    if r/H[p] > q_limit:
                        continue

                print(("Found difference: Particle ID {0:8d}, "+
                        "position {1:14.8e} {2:14.8e}, h = {3:14.8e}, "+
                        "H = {4:14.8e}, dist = {5:14.8e}, dist/H = {6:14.8E}").format(
                            ids[p], pos[p,0], pos[p,1], h[p], H[p], r, r/H[p])
                            )
                print(("                 Neighbour ID {0:8d}, "+
                        "position {1:14.8e} {2:14.8e}, h = {3:14.8e}, "+
                        "H = {4:14.8e}").format(
                            ids[nind], pos[nind,0], pos[nind,1], h[nind], H[nind])
                            )
                print("dwdr py = {0:14.8e}, dwdr swift = {1:14.8e}, 1 - swift/py = {2:12.6f}".format(
                        py, sw, 1 - sw/py ))
                dw = ms.dWdr(r/H[p], H[p])
                dw2 = ms.dWdr(r/H[nind], H[nind])
                found_difference = True

        if do_break and found_difference:
            break

    if not found_difference:
        print("Finished, all the same.")
    if do_break and found_difference:
        quit()





    #------------------------------------------------------
    print("Checking individual gradient contributions")
    #------------------------------------------------------

    found_difference = False
    for p in range(npart):
        for n in range(nneigh_p[p]):
            # all_grads_p has dimensions npart x npart; it is grad_W_j_at_i
            # nids_p[n]-1 is the index we're looking for
            nind = nids_p[p, n]-1
            pyx = all_grads_p[nind, p, 0]
            pyy = all_grads_p[nind, p, 1]
            swx = grads_s[p, 2*n]
            swy = grads_s[p, 2*n+1]

            for P, S in [(pyx, swx), (pyy, swy)]:
                if abs(P) > NULL_SUMS:
                    diff = abs(1 - S/P)
                elif abs(S) > NULL_SUMS:
                    diff = abs(1 - P/S)
                else:
                    continue

                if diff > tolerance:

                    dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1])
                    r = np.sqrt(dx**2 + dy**2)

                    if limit_q:
                        if r/H[p] > q_limit:
                            continue


                    print(("Found difference: ID: {0:14d}  neighbour {1:14d} difference: {7:12.6f} r/H: {6:14.8e}\n"+
                           "                   x: {2:14.8e}         {3:14.8e}\n"+
                           "                   y: {4:14.8e}         {5:14.8e}\n").format(
                                ids[p], nids_p[p,n], pyx, swx, pyy, swy, r/H[p], diff)
                        )
                    found_difference = True
                    if do_break:
                        break
            #  if do_break and found_difference:
            #      break
        if do_break and found_difference:
            break

    if not found_difference:
        print("Finished, all the same.")
    if found_difference and do_break:
        quit()








    #------------------------------------------------------
    print("Checking gradient sums")
    #------------------------------------------------------



    found_difference = False
    for p in range(npart):

        pyx = sum_grad_p[p,0]
        pyy = sum_grad_p[p,1]
        swx = sum_grad_s[p,0]
        swy = sum_grad_s[p,1]

        for P, S in [(pyx, swx), (pyy, swy)]:
            if abs(P) > NULL_SUMS:
                diff = 1 - abs(S/P)
            elif abs(S) > NULL_SUMS:
                diff = 1 - abs(P/S)
            else:
                continue

            if diff > tolerance:

                print(("Found difference: ID: {0:14d}  neighbour {1:14d} difference: {6:12.6f}\n"+
                       "                   x: {2:14.8e}         {3:14.8e}\n"+
                       "                   y: {4:14.8e}         {5:14.8e}\n").format(
                            ids[p], nids_p[p,n], pyx, swx, pyy, swy, diff)
                    )
                found_difference = True
                if do_break:
                    break
        if do_break and found_difference:
            break

    if not found_difference:
        print("Finished, all the same.")
    if do_break and found_difference:
        quit()








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
                if do_break:
                    break
        if do_break and found_difference:
            break

    if not found_difference:
        print("Finished, all the same.")
    if found_difference and do_break:
        quit()










    return






#==========================
def main():
#==========================


    announce()
    extract_gradients_from_snapshot()
    compute_gradients_my_way()
    compare_grads()
    return




if __name__ == '__main__':

    main()

