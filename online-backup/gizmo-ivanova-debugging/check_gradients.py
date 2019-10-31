#!/usr/bin/env python3

#========================================================
# compare results 1:1 from swift and python outputs
# intended to compare Ivanova effective surfaces
#
# You can give the snapshot number as cmd line arg, or
# hard code it.
#
# usage: ./check_gradients.py <dump-number>
#   dump number is optional, is binary dump that 
#   I handwrote for this and implemented into swift
#========================================================


import numpy as np
import pickle
import h5py
import os
import meshless as ms
from read_swift_dumps import extract_dump_data
from my_utils import yesno, one_arg_present

#------------------------
# Filenames
#------------------------

snap = '0001'                   # which snap to use
fname_prefix = 'swift-gizmo-debug-dump_'

# read in cmd line arg snapshot number if present and convert it to formatted string
snap = ms.snapstr(one_arg_present(snap))

srcfile = fname_prefix+snap+'.dat'

swift_dump = 'gizmo-debug-swift-data_'+snap+'.pkl'
part_dump = 'gizmo-debug-swift-particle-data_'+snap+'.pkl'
python_dump = 'gizmo-debug-python-gradient-data_'+snap+'.pkl'


#----------------------
# Behaviour params
#----------------------

tolerance = 1e-2    # relative tolerance threshold for relative float comparison: if (a - b)/a < tolerance, it's fine
NULL = 1e-7         # treat values below this as zeroes
NULL_RELATIVE = 5e-3    # ignore relative values below this threshold


do_break = False    # break after you found a difference

limit_q = True      # whether to ignore differences for high q = r/H; Seems to be stupid round off errors around
q_limit = 0.99      # upper limit for q = r/H if difference is found;

single_particle = False # whether to print out results only for one particle, specified by ID below
single_ID = 516     # ID for which to print out results





#=====================
def announce():
#=====================

    print("CHECKING GRADIENTS.")
    print("tolerance is set to:", tolerance)
    print("NULL is set to:", NULL)
    print("NULL_RELATIVE is set to:", NULL_RELATIVE)
    print("do break if found difference?", do_break)
    print("use upper limit for q = r/H?", limit_q)
    if limit_q:
        print("upper limit for q = r/H is:", q_limit)
    if single_particle:
        print("Printing output only for ID", single_ID)
    print("---------------------------------------------------------")
    print()

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

    print("Computing Gradients")

    part_filep = open(part_dump, 'rb')
    data_part = pickle.load(part_filep)
    ids, pos, h = data_part

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    npart = x.shape[0]

    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)

    # set up such that you don't need arguments in functions any more
    h = H
    fact = 1
    L = 1
    periodic = True
    kernel = 'cubic_spline'

    # first get neighbour data
    neighbour_data = ms.get_neighbour_data_for_all(x, y, h, fact=fact, L=L, periodic=periodic)

    maxneigh = neighbour_data.maxneigh
    neighbours = neighbour_data.neighbours
    nneigh = neighbour_data.nneigh
    iinds = neighbour_data.iinds

    # compute all psi_j(x_i) for all i, j
    # first index: index j of psi: psi_j(x)
    # second index: index of x_i: psi(x_i)

    psi_j_at_i = np.zeros((npart, maxneigh), dtype=np.float)
    omega = np.zeros(npart, dtype=np.float)

    for j in range(npart):
        for i, ind_n in enumerate(neighbours[j]):
            # kernels are symmetric in x_i, x_j, but h can vary!!!!
            psi_j_at_i[j, i] = ms.psi(x[ind_n], y[ind_n], x[j], y[j], h[ind_n], 
                                    kernel=kernel, fact=fact, L=L, periodic=periodic)
            omega[ind_n] += psi_j_at_i[j, i]

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

    for i in range(npart):
        for j, jind in enumerate(neighbours[i]):
            dx, dy = ms.get_dx( x[i], x[jind], y[i], y[jind], L=L, periodic=periodic)
            r = np.sqrt(dx**2 + dy**2)

            iind = iinds[i, j]
            dw = ms.dWdr(r/H[i], H[i], kernel)

            grad_W_j_at_i[jind, iind, 0] = dw * dx / r
            grad_W_j_at_i[jind, iind, 1] = dw * dy / r

            sum_grad_W[i] += grad_W_j_at_i[jind, iind]

            # store other stuff
            #  dwdr[jind, iind] = dw
            #  r_store[jind, iind] = r
            #  dx_store[jind, iind] = dx
            #  dx_store[jind, iind] = dy
            dwdr[i, j] = dw
            r_store[i, j] = r
            dx_store[i, j, 0] = dx
            dx_store[i, j, 1] = dy




    # finish computing the gradients: Need W(r, h), which is currently stored as psi
    for j in range(npart):
        for i, ind_n in enumerate(neighbours[j]):
            grad_psi_j_at_i[j, i, 0] = grad_W_j_at_i[j, i, 0]/omega[ind_n] - psi_j_at_i[j, i] * sum_grad_W[ind_n, 0]/omega[ind_n]**2
            grad_psi_j_at_i[j, i, 1] = grad_W_j_at_i[j, i, 1]/omega[ind_n] - psi_j_at_i[j, i] * sum_grad_W[ind_n, 1]/omega[ind_n]**2
                #  grad_psi_j_at_i[j, iind, :] = grad_W_j_at_i[j, iind, :]/omega[iind] - psi_j_at_i[j, iind] * sum_grad_W[iind, :]/omega[iind]**2






    nneighs = np.array([len(n) for n in neighbours], dtype=np.int)
    maxlen = np.max(nneighs)
    nids = np.zeros((npart, maxlen), dtype=np.int)

    for nb in range(npart):
        nids[nb, :nneighs[nb]] = ids[neighbours[nb]]




    data_dump = [grad_psi_j_at_i, sum_grad_W, grad_W_j_at_i, dwdr, nids, nneighs, omega, r_store, dx_store, iinds]
    dumpfile = open(python_dump, 'wb')
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped python data")

    return










#========================================
def compute_gradients_my_way_old():
#========================================
    """
    Compute gradients using my python module, and dump results in a pickle
    """


    if os.path.isfile(python_dump):
        if not yesno("Dump file", python_dump, "already exists. Shall I overwrite it?"):
            return

    part_filep = open(part_dump, 'rb')
    data_part = pickle.load(part_filep)
    ids, pos, h = data_part

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    npart = x.shape[0]

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
    r_store = np.zeros((npart, npart), dtype=np.float)
    dx_store = np.zeros((npart, npart, 2), dtype=np.float)


    GIZMO_ZERO = 1e-10
    for i in range(npart):
        for j in neighbours[i]:
            # get kernel gradients

            dx, dy = ms.get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)

            # set dx, dy explicitly to zero
            if abs(dx) < GIZMO_ZERO : dx = 0.0
            if abs(dy) < GIZMO_ZERO : dy = 0.0

            r = np.sqrt(dx**2 + dy**2)
            if r != 0:
                dwdr[j,i] = ms.dWdr(r/h[i], h[i], kernel)
                r_store[i,j] = r
                dx_store[i,j,0] = dx # store i - j, not j - i
                dx_store[i,j,1] = dy
                grad_W_j_at_i[j, i, 0] = dwdr[j,i] * dx / r
                grad_W_j_at_i[j, i, 1] = dwdr[j,i] * dy / r
            # else: zero anyway
            #  else:
            #      ms.grad_W_k_at_l[k, l, 0] = 0
            #      ms.grad_W_k_at_l[k, l, 1] = 0
            


    sum_grad_W = np.zeros((npart, 2), dtype=np.float)

    for i in range(npart):
        # you can skip the self contribution here, the gradient at r = 0 is 0
        # sum along fixed/same h_i
        sum_grad_W[i] = np.sum(grad_W_j_at_i[neighbours[i], i], axis=0)



    nneighs = np.array([len(n) for n in neighbours], dtype=np.int)
    maxlen = np.max(nneighs)
    nids = np.zeros((npart, maxlen), dtype=np.int)

    for nb in range(npart):
        nids[nb, :nneighs[nb]] = ids[neighbours[nb]]




    data_dump = [sum_grad_W, grad_W_j_at_i, dwdr, nids, nneighs, omega, r_store, dx_store]
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

    # NOTE: THE INDEX CALLING FOR PYTHON AND SWIFT DATA ONLY WORKS
    # BECAUSE THE PARTICLE DATA ARE READ IN FROM SWIFT DUMPS, WHICH ARE ALREADY SORTED

    swift_filep = open(swift_dump, 'rb')
    python_filep = open(python_dump, 'rb')
    part_filep = open(part_dump, 'rb')

    data_swift = pickle.load(swift_filep)
    grads_s, grads_contrib_s, sum_grad_s, dwdr_s, nids_s, nneigh_s, omega_s, vol_s, dx_s, r_s, nneigh_Aij_s, nids_Aij_s, Aij_s = data_swift

    data_part = pickle.load(part_filep)
    ids, pos, h = data_part
    H = ms.get_H(h)

    data_python = pickle.load(python_filep)
    grads_p, sum_grad_p, grads_contrib_p, dwdr_p, nids_p, nneigh_p, omega_p, r_p, dx_p, iinds = data_python

    #  iinds:         iinds[i, j] = which index does particle i have in the neighbour
    #                      list of particle j, where j is the j-th neighbour of i
    #                      Due to different smoothing lengths, particle j can be the
    #                      neighbour of i, but i not the neighbour of j.
    #                      In that case, the particles will be assigned indices j > nneigh[i]


    swift_filep.close()
    python_filep.close()
    part_filep.close()


    npart = ids.shape[0]





    #----------------------------------------------
    def check_number_of_neighbours():
    #----------------------------------------------

        print("Checking number of neighbours")

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
    def check_neighbour_IDs():
    #----------------------------------------------

        print("Checking neighbour IDs")

        found_difference = False
        for p in range(npart):
            pyarr = nids_p[p][:nneigh_p[p]]
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
    def check_vol():
    #------------------------------------------------------


        print("Checking volumes and normalizations")

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
                    if single_particle and ids[p] != single_ID:
                        continue
                    print("Found difference: ID", ids[p], "v:", pyv, swv, "n:", pyn, swn)
                    if do_break:
                        break
            if do_break and found_difference:
                break

        if not found_difference:
            print("Finished, all the same.")
        if found_difference and do_break:
            quit()






    #------------------------------------------------------
    def check_dwdr():
    #------------------------------------------------------

        print("Checking radial gradients")

        found_difference = False
        for p in range(npart):
            for n in range(nneigh_p[p]):
                py = dwdr_p[p, n]
                sw = dwdr_s[p, n]

                if abs(py) > NULL:
                    diff = abs(1 - sw/py)
                elif abs(sw) > NULL:
                    diff = abs(1 - py/sw)
                else:
                    continue

                if diff > tolerance:

                    if single_particle and ids[p] != single_ID:
                        continue

                    nind = nids_p[p, n]-1
                    dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1])
                    r = np.sqrt(dx**2 + dy**2)

                    if limit_q:
                        if r/H[p] > q_limit:
                            continue

                    print("Found difference in radial gradients:")
                    print((" Particle ID {0:8d}, "+
                            "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                            "H = {4:14.7e}, dist = {5:14.7e}, dist/H = {6:14.7e}").format(
                                ids[p], pos[p,0], pos[p,1], h[p], H[p], r, r/H[p])
                                )
                    print(("Neighbour ID {0:8d}, "+
                            "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                            "H = {4:14.7e}").format(
                                ids[nind], pos[nind,0], pos[nind,1], h[nind], H[nind])
                                )
                    print("dwdr py = {0:14.7e}, dwdr swift = {1:14.7e}, diff = {2:12.6f}".format(
                            py, sw, diff ))
                    dw = ms.dWdr(r/H[p], H[p])
                    dw2 = ms.dWdr(r/H[n], H[n])
                    found_difference = True

            if do_break and found_difference:
                break

        if not found_difference:
            print("Finished, all the same.")
        if do_break and found_difference:
            quit()





    #------------------------------------------------------
    def check_r():
    #------------------------------------------------------

        print("Checking r")

        found_difference = False
        for p in range(npart):
            for n in range(nneigh_p[p]):
                py = r_p[p, n]
                sw = r_s[p, n]

                if abs(py) > NULL:
                    diff = abs(1 - sw/py)
                elif abs(sw) > NULL:
                    diff = abs(1 - py/sw)
                else:
                    print("Skipping", py, sw)
                    continue

                if diff > tolerance:
                    if single_particle and ids[p] != single_ID:
                        continue

                    nind = nids_p[p, n]-1
                    dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1])
                    r = np.sqrt(dx**2 + dy**2)

                    if limit_q:
                        if r/H[p] > q_limit:
                            continue

                    print("Found difference: in r:")
                    print((" Particle ID {0:8d}, "+
                            "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                            "H = {4:14.7e}, dist = {5:14.7e}, dist/H = {6:14.7e}").format(
                                ids[p], pos[p,0], pos[p,1], h[p], H[p], r, r/H[p])
                                )
                    print(("Neighbour ID {0:8d}, "+
                            "position {1:14.7e} {2:14.7e}, h = {3:14.7e}, "+
                            "H = {4:14.7e}").format(
                                ids[nind], pos[nind,0], pos[nind,1], h[nind], H[nind])
                                )
                    print("r py = {0:14.7e}, r swift = {1:14.7e}, 1 - swift/py = {2:12.6f}".format(
                            py, sw, 1 - sw/py ))
                    print(r)
                    found_difference = True

            if do_break and found_difference:
                break

        if not found_difference:
            print("Finished, all the same.")
        if do_break and found_difference:
            quit()





    #------------------------------------------------------
    def check_dx():
    #------------------------------------------------------

        print("Checking dx")

        found_difference = False
        for p in range(npart):
            for n in range(nneigh_p[p]):
                pyx = dx_p[p, n, 0]
                pyy = dx_p[p, n, 1]
                swx = dx_s[p, 2*n]
                swy = dx_s[p, 2*n+1]

                if abs(pyx) > NULL:
                    if abs(pyy/pyx) < NULL_RELATIVE:
                        continue
                if abs(pyy) > NULL:
                    if abs(pyx/pyy) < NULL_RELATIVE:
                        continue


                for P, S in [(pyx, swx), (pyy, swy)]:
                    if abs(P) > NULL:
                        diff = abs(1 - S/P)
                    elif abs(S) > NULL:
                        diff = abs(1 - P/S)
                    else:
                        continue



                    if diff > tolerance:
                        if single_particle and ids[p] != single_ID:
                            continue

                        nind = nids_p[p, n]-1
                        dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1])
                        r = np.sqrt(dx**2 + dy**2)

                        if limit_q:
                            if r/H[p] > q_limit:
                                continue


                        print("In dx:")
                        print("Found difference:  ID: {0:14d} neighbour ID: {1:14d}".format(ids[p], nids_p[p,n]))
                        #  print("Found difference:  ID: {0:14d} neighbour ID: {1:14d} difference: {2:12.6f} r/H: {3:14.7e}".format(ids[p], nids_p[p,n], diff, r/H[p]))
                        print("       positions:   x:  {0:14.7e}  {1:14.7e}".format(pos[p][0], pos[nind][0]))
                        print("       positions:   y:  {0:14.7e}  {1:14.7e}".format(pos[p][1], pos[nind][1]))
                        print("       dx:         python:                      swift:                    python/swift:")
                        print("                   x: {0:14.7e}           {1:14.7e}            {2:14.7e}".format(pyx, swx, pyx/swx))
                        print("                   y: {0:14.7e}           {1:14.7e}            {2:14.7e}".format(pyy, swy, pyy/swy))
                        print()

                        found_difference = True
                        if do_break:
                            break
                if do_break and found_difference:
                    break
            if do_break and found_difference:
                break

        if not found_difference:
            print("Finished, all the same.")
        if found_difference and do_break:
            quit()







    #------------------------------------------------------
    def check_individual_contributions():
    #------------------------------------------------------

        print("Checking individual gradient contributions")

        found_difference = False
        for p in range(npart):
            for n in range(nneigh_p[p]):
                nind = nids_p[p,n]-1
                iind = iinds[p,n]
                pyx = grads_contrib_p[nind, iind, 0]
                pyy = grads_contrib_p[nind, iind, 1]
                swx = grads_contrib_s[p, 2*n]
                swy = grads_contrib_s[p, 2*n+1]

                if abs(pyx) > NULL:
                    if abs(pyy/pyx) < NULL_RELATIVE:
                        continue
                if abs(pyy) > NULL:
                    if abs(pyx/pyy) < NULL_RELATIVE:
                        continue


                for P, S in [(pyx, swx), (pyy, swy)]:
                    if abs(P) > NULL:
                        diff = abs(1 - S/P)
                    elif abs(S) > NULL:
                        diff = abs(1 - P/S)
                    else:
                        continue


                    if diff > tolerance:
                        if single_particle and ids[p] != single_ID:
                            continue

                        nind = nids_p[p, n]-1
                        dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1])
                        r = np.sqrt(dx**2 + dy**2)

                        if limit_q:
                            if r/H[p] > q_limit:
                                continue


                        print("In individual gradient contribution:")
                        print("Found difference:     ID: {0:14d}  neighbour {1:14d} difference: {2:12.6f} r/H: {3:14.7e}\n".format(ids[p], nids_p[p,n], diff, r/H[p]))
                        print("                           python:                  swift:                      python/swift:")
                        print(" gradient contribution x: {0:14.7e}           {1:14.7e}          {2:14.7f}".format(pyx, swx, pyx/swx))
                        print(" gradient contribution y: {0:14.7e}           {1:14.7e}          {2:14.7f}".format(pyy, swy, pyy/swy)) 

                        P = dwdr_p[p, n]; S = dwdr_s[p, n];
                        print("                    dwdr: {0:14.7e}           {1:14.7e}          {2:14.7f}".format(P, S, P/S))
                        P = r_p[p, n]; S = r_s[p, n];
                        print("                       r: {0:14.7e}           {1:14.7e}          {2:14.7f}".format(P, S, P/S))
                        P = dx_p[p, n, 0]; S = dx_s[p, 2*n];
                        print("                      dx: {0:14.7e}           {1:14.7e}          {2:14.7f}".format(P, S, P/S))
                        P = dx_p[p, n, 1]; S = dx_s[p, 2*n+1];
                        print("                      dy: {0:14.7e}           {1:14.7e}          {2:14.7f}".format(P, S, P/S))
                        print()
                        found_difference = True
                        if do_break:
                            break
                if do_break and found_difference:
                    break
            if do_break and found_difference:
                break

        if not found_difference:
            print("Finished, all the same.")
        if found_difference and do_break:
            quit()







    #------------------------------------------------------
    def check_grad_sums():
    #------------------------------------------------------

        print("Checking gradient sums")

        found_difference = False
        for p in range(npart):

            pyx = sum_grad_p[p,0]
            pyy = sum_grad_p[p,1]
            swx = sum_grad_s[p,0]
            swy = sum_grad_s[p,1]


            # find maximal contribution so you can estimate relative 
            # tolerance

            pymax = np.absolute(grads_contrib_p[p, :nneigh_p[p]]).max()
            swmax = np.absolute(grads_contrib_s[:2*nneigh_p[p]+1]).max()

            allmax = max(pymax, swmax)

            TEMPNULL = allmax*NULL_RELATIVE

            for P, S in [(pyx, swx), (pyy, swy)]:
                if abs(P) > TEMPNULL:
                    diff = 1 - abs(S/P)
                elif abs(S) > TEMPNULL:
                    diff = 1 - abs(P/S)
                else:
                    continue

                if diff > tolerance:
                    if single_particle and ids[p] != single_ID:
                        continue
                    
                    print("In gradient sums:")
                    print("Found difference: ID: {0:14d} difference: {1:12.6f}".format(ids[p], diff))
                    print("                       python:                  swift:                      python/swift:")
                    print("              sum  x: {0:14.7e}         {1:14.7e}         {2:14.7f}".format(pyx, swx, pyx/swx))
                    print("              sum  y: {0:14.7e}         {1:14.7e}         {2:14.7f}".format(pyy, swy, pyy/swy))
                        
                    print("Values are ignored below {0:14.7e}\n".format(TEMPNULL))

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
    def check_final_grads():
    #------------------------------------------------------

        print("Checking final gradients")

        found_difference = False
        for p in range(npart):
            add = 0
            for n in range(nneigh_p[p]):

                ns = n+add
                nID_p = nids_p[p][n]
                nID_s = nids_Aij_s[p][ns]

                while nID_p > nID_s:
                    # From the SWIFT output, there may be more neighbours than from python
                    # if rij < H_j but r_ij > H_i, particle j has i as neighbour, but
                    # i hasn't got j as neighbour. Both neighbours will be written down though
                    add += 1
                    try:
                        nID_s = nids_Aij_s[p][n+add]
                    except IndexError:
                        print("Something fucky going on")
                        print("particle:", ids[p], "neighbour swift:", nID_s, "neighbour py:", nID_p, "p:", p, "n:", n)
                        print("nneigh py:", nneigh_p[p], "nneigh sw:", nneigh_Aij_s[p])
                        quit()

                ns = n+add


                i = iinds[p, n]
                j = nids_p[p, n]-1
                pyx = grads_p[j, i, 0]
                pyy = grads_p[j, i, 1]
                swx = grads_s[p, 2*ns]
                swy = grads_s[p, 2*ns+1]

                maxgrad = np.absolute(grads_p[p]).max()

                for P, S in [(pyx, swx), (pyy, swy)]:
                    if abs(P) > NULL:
                        diff = abs(1 - S/P)
                    elif abs(S) > NULL:
                        diff = abs(1 - P/S)
                    else:
                        continue

                    if P < maxgrad*NULL_RELATIVE:
                        continue

                    if diff > tolerance:
                        if single_particle and ids[p] != single_ID:
                            continue

                        nind = nids_p[p, n]-1
                        dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1])
                        r = np.sqrt(dx**2 + dy**2)

                        if limit_q:
                            if r/H[p] > q_limit:
                                continue


                        print("In final gradients:")
                        print("Found difference:  ID: {0:14d}           neighbour ID: {1:14d}".format(ids[p], nids_p[p,n]))
                        #  print("Found difference:  ID: {0:14d} neighbour ID: {1:14d} difference: {2:12.6f} r/H: {3:14.7e}".format(ids[p], nids_p[p,n], diff, r/H[p]))
                        print("       positions:   x:  {0:14.4f}  {1:14.4f}".format(pos[p][0], pos[nind][0]))
                        print("       positions:   y:  {0:14.4f}  {1:14.4f}".format(pos[p][1], pos[nind][1]))
                        print("       del psi/del x:  python:                  swift:                    python/swift:")
                        print("                   x: {0:14.7e}           {1:14.7e}            {2:14.7e}".format(pyx, swx, pyx/swx))
                        print("                   y: {0:14.7e}           {1:14.7e}            {2:14.7e}".format(pyy, swy, pyy/swy))
                        print()

                        found_difference = True
                        if do_break:
                            break
                if do_break and found_difference:
                    break
            if do_break and found_difference:
                break

        if not found_difference:
            print("Finished, all the same.")
        if found_difference and do_break:
            quit()






    #==================================
    # Do the actual checks
    #==================================

    check_number_of_neighbours()
    check_neighbour_IDs()
    check_vol()
    check_dwdr()
    check_r()
    check_dx()
    check_individual_contributions()
    check_grad_sums()
    check_final_grads()

    return






#==========================
def main():
#==========================

    announce()
    extract_dump_data(srcfile, swift_dump, part_dump)
    compute_gradients_my_way()
    compare_grads()
    return




if __name__ == '__main__':

    main()

