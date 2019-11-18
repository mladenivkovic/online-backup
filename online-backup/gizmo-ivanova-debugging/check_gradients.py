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
import os

import meshless as ms
from my_utils import yesno, one_arg_present

from read_swift_dumps import extract_dump_data
from filenames import get_srcfile, get_dumpfiles
from compute_gradients import compute_gradients_my_way

#------------------------
# Filenames and global stuff
#------------------------

periodic = True
srcfile = get_srcfile()
swift_dump, part_dump, python_surface_dump, python_grad_dump = get_dumpfiles()


#----------------------
# Behaviour params
#----------------------

tolerance = 1e-3    # relative tolerance threshold for relative float comparison: if (a - b)/a < tolerance, it's fine
NULL = 1e-6         # treat values below this as zeroes
NULL_RELATIVE = 5e-3    # ignore relative values below this threshold


do_break = True    # break after you found a difference

limit_q = True      # whether to ignore differences for high q = r/H; Seems to be stupid round off errors around
q_limit = 0.99      # upper limit for q = r/H if difference is found;

single_particle = False # whether to print out results only for one particle, specified by ID below
single_ID = 4463     # ID for which to print out results




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











#================================
def compare_grads():
#================================
    """
    compare the gradients you got
    """

    # NOTE: THE INDEX CALLING FOR PYTHON AND SWIFT DATA ONLY WORKS
    # BECAUSE THE PARTICLE DATA ARE READ IN FROM SWIFT DUMPS, WHICH ARE ALREADY SORTED

    swift_filep = open(swift_dump, 'rb')
    grads_s         = pickle.load(swift_filep)
    grads_contrib_s = pickle.load(swift_filep)
    sum_grad_s      = pickle.load(swift_filep)
    dwdr_s          = pickle.load(swift_filep)
    Wjxi_s          = pickle.load(swift_filep)
    nids_s          = pickle.load(swift_filep)
    nneigh_s        = pickle.load(swift_filep)
    omega_s         = pickle.load(swift_filep)
    vol_s           = pickle.load(swift_filep)
    dx_s            = pickle.load(swift_filep)
    r_s             = pickle.load(swift_filep)
    nneigh_Aij_s    = pickle.load(swift_filep)
    nids_Aij_s      = pickle.load(swift_filep)
    Aij_s           = pickle.load(swift_filep)
    swift_filep.close()

    part_filep = open(part_dump, 'rb')
    ids     = pickle.load(part_filep)
    pos     = pickle.load(part_filep)
    h       = pickle.load(part_filep)
    part_filep.close()
    H = ms.get_H(h)

    python_filep = open(python_grad_dump, 'rb')
    grads_p         = pickle.load(python_filep)
    sum_grad_p      = pickle.load(python_filep)
    grads_contrib_p = pickle.load(python_filep)
    dwdr_p          = pickle.load(python_filep)
    Wjxi_p          = pickle.load(python_filep)
    nids_p          = pickle.load(python_filep)
    nneigh_p        = pickle.load(python_filep)
    omega_p         = pickle.load(python_filep)
    r_p             = pickle.load(python_filep)
    dx_p            = pickle.load(python_filep)
    iinds           = pickle.load(python_filep)
    #  iinds:         iinds[i, j] = which index does particle i have in the neighbour
    #                      list of particle j, where j is the j-th neighbour of i
    #                      Due to different smoothing lengths, particle j can be the
    #                      neighbour of i, but i not the neighbour of j.
    #                      In that case, the particles will be assigned indices j > nneigh[i]
    python_filep.close()

    L = ms.read_boxsize()



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

                # apparently due to roundoff errors python doesn't always find 
                # the same number of neighbours. If tolerance is small enough,
                # remove the neighbour and its data from swift data.
                # this means that py > sw
                if py < sw:
                    pyn = nids_p[p][:nneigh_p[p]]
                    swn = nids_s[p][:nneigh_s[p]]
                    to_remove = []
                    for n in range(nneigh_s[p]):
                        if swn[n] not in pyn:
                            nind = swn[n] - 1
                            dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)
                            r = np.sqrt(dx**2 + dy**2)
                            if r/H[p] > 1.0 and r/H[p] < 1+1e-3:
                                print("removing neighbour {0:7d} from particle {1:7d}. r/H = {2:14.7f}".format(ids[nind], ids[p], r/H[p]))
                                to_remove.append(n)

                    if len(to_remove) > 0:
                        for i, N in enumerate(to_remove):
                            n = N-i # reduce index because you remove stuff
                            print(n, N)
                            print(nids_s[p][:nneigh_s[p]])
                            nids_s[p][n:nneigh_s[p]-1] = nids_s[p][n+1:nneigh_s[p]]
                            nneigh_s[p]-=1
                            print(nids_s[p][:nneigh_s[p]])
                            print()


            
            if py != sw:

                if py > sw:
                    larger_name = 'python'
                    larger = nids_p[p]
                    nl = nneigh_p[p]
                    smaller = nids_s[p]
                    ns = nneigh_s[p]
                else:
                    larger_name = 'swift'
                    smaller = nids_p[p]
                    ns = nneigh_p[p]
                    larger = nids_s[p]
                    nl = nneigh_s[p]

                #  print("Larger is:", larger_name)
                to_remove = []
                for n in range(nl):
                    if larger[n] not in smaller:
                        nind = larger[n] - 1
                        dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)
                        r = np.sqrt(dx**2 + dy**2)

                        if abs(1 - r/H[p]) < 1e-3:
                            print("removing neighbour {0:7d} from particle {1:7d}. r/H = {2:14.7f}, dataset={3:10}".format(ids[nind], ids[p], r/H[p], larger_name))
                            to_remove.append(n)

                if len(to_remove) > 0:
                    for i, N in enumerate(to_remove):
                        n = N-i # reduce index because you remove stuff
                        larger[n:nl-1] = larger[n+1:nl]

                        if larger_name == 'swift':
                            nneigh_s[p] -= 1
                            for arr in [dwdr_s, r_s, Wjxi_s]:
                                arr[p, n:nl-1] = arr[p, n+1:nl]
                            for arr in [dx_s, grads_s]:
                                arr[p, 2*n:2*(nl-1)] = arr[p, 2*n+2:2*nl]

                        else:
                            nneigh_p[p] -= 1
                            for arr in [dwdr_p, r_p, dx_p, Wjxi_p, iinds]:
                                arr[p, n:nl-1] = arr[p, n+1:nl]

                        nl-=1


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
    def check_wjxi():
    #------------------------------------------------------

        print("Checking Kernel Values")

        found_difference = False
        for p in range(npart):
            for n in range(nneigh_p[p]):
                py = Wjxi_p[p,n]
                sw = Wjxi_s[p,n]

                maxguess = Wjxi_s[p].max()
                TEMPNULL = maxguess * NULL_RELATIVE

                if py > TEMPNULL:
                    diff = abs(1 - sw/py)
                elif sw > TEMPNULL:
                    diff = abs(1 - py/sw)
                else:
                    continue

                if diff > tolerance:

                    if single_particle and ids[p] != single_ID:
                        continue
                    nind = nids_p[p, n]-1

                    dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)
                    r = np.sqrt(dx**2 + dy**2)

                    if limit_q:
                        if r/H[p] > q_limit:
                            continue

                    print("Found difference in kernel values:")
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
                    print("Wj(xi) py = {0:14.7e}, Wj(xi) swift = {1:14.7e}, diff = {2:12.6f}".format(
                            py, sw, diff ))
                    print("TEMPNULL is", TEMPNULL) 
                    print("index guessing ", py, Wjxi_p[p,n], ms.W(r/H[p], H[p]))
                    print("p is", p, "n is", n, "nind is", nind)
                    print("recompute hi: ", ms.W(r/H[nind], H[nind]))
                    print("recompute hj: ", ms.W(r/H[p], H[p]))
                    print("swn:", nids_s[p, :nneigh_s[p]])
                    print("pyn:", nids_p[p, :nneigh_p[p]])
                    print("dx:", dx, "dy", dy, "r", r)
                    print("Ncheck", nids_s[p, n], nids_p[p,n])
                    print()
                    found_difference = True

                    if do_break and found_difference: # neighbour loop
                        break
            if do_break and found_difference:   # particle loop
                break

        if not found_difference:
            print("Finished, all the same.")
        if do_break and found_difference:
            quit()








    #------------------------------------------------------
    def check_vol():
    #------------------------------------------------------


        print("Checking volumes and normalizations")

        found_difference = False
        counter = 0
        for p in range(npart):

            pyv = 1/omega_p[p]
            pyn = omega_p[p]
            swv = vol_s[p]
            swn = omega_s[p]

            #  if swv < pyv:
            #      if abs(1- swv/pyv) > 1e-4:
            #          counter +=1
            #          print("Got swift volume > python; ID", ids[p], pyv, swv, abs(1-swv/pyv))

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
                    print("In Volumes and Normalizations:")
                    print("Found difference: ID", ids[p])
                    print("            python          swift               |1 - py/sw|")
                    print("volume:    {0:14.7e}  {1:14.7e}  {2:14.7f}".format(pyv, swv, abs(1-pyv/swv)))
                    print("norm:      {0:14.7e}  {1:14.7e}  {2:14.7f}".format(pyn, swn, abs(1-pyn/swn)))

                    found_difference = True
                    break

            if do_break and found_difference:
                break
        print("Volume sums:")
        print("Python: {0:14.7f}".format(np.sum(1./omega_p)))
        print("Swift:  {0:14.7f}".format(np.sum(vol_s)))
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
                    dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)

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
                    dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)

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
            dxmax = np.absolute(dx_p[p][:nneigh_p[p]]).max()
            for n in range(nneigh_p[p]):
                pyx = dx_p[p, n, 0]
                pyy = dx_p[p, n, 1]
                swx = dx_s[p, 2*n]
                swy = dx_s[p, 2*n+1]

                if abs(pyx) < NULL_RELATIVE or abs(pyy) < NULL_RELATIVE:
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
                        dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)

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
            gradmax = np.absolute(grads_contrib_p[:nneigh_p[p]]).max()
            gradmin = gradmax * NULL_RELATIVE
            for n in range(nneigh_p[p]):
                nind = nids_p[p,n]-1
                iind = iinds[p,n]
                pyx = grads_contrib_p[nind, iind, 0]
                pyy = grads_contrib_p[nind, iind, 1]
                swx = grads_contrib_s[p, 2*n]
                swy = grads_contrib_s[p, 2*n+1]

                if abs(pyx) < gradmin or abs(pyy) < gradmin:
                    continue
                #  if abs(pyx) > NULL:
                #      if abs(pyy/pyx) < NULL_RELATIVE:
                #          continue
                #  if abs(pyy) > NULL:
                #      if abs(pyx/pyy) < NULL_RELATIVE:
                #          continue


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
                        dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)
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
                        print("Ignoring values below {0:14.7e}".format(gradmin))
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
                    if abs(P) < maxgrad*NULL_RELATIVE:
                        continue
 
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
                        dx, dy = ms.get_dx(pos[p,0], pos[nind,0], pos[p,1], pos[nind,1], L=L, periodic=periodic)
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
                        print("                   x: {0:14.7e}           {1:14.7e}            {2:14.7f}".format(pyx, swx, pyx/swx))
                        print("                   y: {0:14.7e}           {1:14.7e}            {2:14.7f}".format(pyy, swy, pyy/swy))
                        print()

                        py = dwdr_p[p, n]
                        sw = dwdr_s[p, n]
                        print("   dwdr:        {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = r_p[p, n]
                        sw = r_s[p, n]
                        print("   r:           {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = dx_p[p, n, 0]
                        sw = dx_s[p, 2*n]
                        print("   dx[0]:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = dx_p[p, n, 1]
                        sw = dx_s[p, 2*n+1]
                        print("   dx[1]:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        nind = nids_p[p, n]-1
                        py = sum_grad_p[nind,0]
                        sw = sum_grad_s[nind,0]
                        print("   sum_grad[0]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = sum_grad_p[nind,1]
                        sw = sum_grad_s[nind,1]
                        print("   sum_grad[1]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))
                        print()
                        print("Values are ignored below {0:14.7e}".format(maxgrad*NULL_RELATIVE))

                        #  Vp = 1/omega_p[p]
                        #  nind = nids_p[p,n]-1
                        #  xi = pos[p,0]
                        #  yi = pos[p,1]
                        #  xj = pos[nind, 0]
                        #  yj = pos[nind, 1]
                        #  W_j_at_i = ms.psi(xi, yi, xj, yj, H[nind])
                        #
                        #  first = 1/omega_p[p] * dwdr_p[p,n] * dx_p[p,n,0] / r
                        #  second = W_j_at_i * sum_grad_p[nind, 0] / omega_p[nind]**2
                        #
                        #
                        #  #  i = iinds[p, n]
                        #  #  j = nids_p[p, n]-1
                        #  i = n
                        #  j = p
                        #  pyx = grads_p[j, i, 0]
                        #  swx = grads_s[p, 2*ns]
                        #  print(" testing x: {0:14.7f}  {1:14.7f}".format(first+second, first-second ))
                        #  print(" testing x: {0:14.7f}  {1:14.7f}".format(-first+second, -first-second ))
                        #  print(" compare x: {0:14.7f}  {1:14.7f}".format(pyx, swx))
                        #  print("Indices used:",p, n, nids_p[p,n]-1, iinds[p,n])
                        #  print("?? {0:14.7f}".format(sum_grad_p[nids_p[p,n]-1, 0]))

 

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

    #  check_number_of_neighbours()
    #  check_neighbour_IDs()
    check_wjxi()
    check_vol()
    check_dwdr()
    check_r()
    check_dx()
    check_individual_contributions()
    #  check_grad_sums()
    #  check_final_grads()

    return






#==========================
def main():
#==========================

    announce()
    extract_dump_data()
    compute_gradients_my_way(periodic)
    compare_grads()
    return




if __name__ == '__main__':

    main()

