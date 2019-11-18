#!/usr/bin/env python3

#========================================================
# compare results 1:1 from swift and python outputs
# intended to compare Ivanova effective surfaces
#
# CALL check_gradients.py FIRST!!!! THIS SCRIPT NEEDS
# THE OUTPUT THAT check_gradients.py PRODUCES!
# usage: ./compare_Aijs.py <dump-number>
#   dump number is optional, is binary dump that 
#   I handwrote for this and implemented into swift
#========================================================


import numpy as np
import pickle
import h5py
import os

import meshless as ms
from my_utils import yesno, one_arg_present

from filenames import get_srcfile, get_dumpfiles
from read_swift_dumps import extract_dump_data

#-------------------------------
# Filenames and global stuff
#-------------------------------

periodic=True
srcfile = get_srcfile()
swift_dump, part_dump, python_surface_dump, python_grad_dump = get_dumpfiles()


#----------------------
# Behaviour params
#----------------------

tolerance = 1e-1    # relative tolerance threshold for relative float comparison: if (a - b)/a < tolerance, it's fine
NULL_RELATIVE = 1e-5 # relative tolerance for values to ignore below this value






#=====================
def announce():
#=====================

    print("CHECKING SURFACES.")
    print("tolerance is set to:", tolerance)
    print("NULL_RELATIVE is set to:", NULL_RELATIVE)
    print("---------------------------------------------------------")
    print()

    return











#========================================
def compute_Aij_my_way():
#========================================
    """
    Compute Aij using my python module, and dump results in a pickle
    """


    if os.path.isfile(python_surface_dump):
        if not yesno("Dump file", python_surface_dump, "already exists. Shall I overwrite it?"):
            return


    print("Computing Aij")

    part_filep = open(part_dump, 'rb')
    ids     = pickle.load(part_filep)
    pos     = pickle.load(part_filep)
    h       = pickle.load(part_filep)
    part_filep.close()
    H = ms.get_H(h)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    npart = x.shape[0]

    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)
    m = 1
    rho = 1
    L = ms.read_boxsize()

    A_ij_all, neighbours_all = ms.Aij_Ivanova_all(x, y, H, m, rho, L=L)
   
    Aijs = np.zeros((npart, 200, 2), dtype=np.float)
    nneighs = np.zeros((npart), dtype=np.int)
    neighbour_ids = np.zeros((npart, 200), dtype=np.int)

    inds = np.argsort(ids)

    for i, ind in enumerate(inds):
        # i: index in arrays to write
        # ind: sorted index
        nbs = np.array(neighbours_all[ind])               # list of neighbours of current particle
        nneighs[i] = nbs.shape[0]
        ninds = np.argsort(ids[nbs])  # indices of neighbours in nbs array sorted by IDs
        #  ninds = np.argsort(np.array(ids[nbs]))  # indices of neighbours in nbs array sorted by IDs

        for n in range(nneighs[i]):
            nind = nbs[ninds[n]]                # index of n-th neighbour to write in increasing ID order in global arrays 
            neighbour_ids[i,n] = ids[nind]
            Aijs[i,n] = A_ij_all[ind, ninds[n]]


    

    #  for i in inds[:5]:
    #
    #      #  print("ID: {0:8d} ||".format(ids[inds[i]]), end='')
    #      print("ID: {0:8d} {1:8d} ||".format(ids[i], nneighs[i]))
    #
    #      for n in range(nneighs[i]):
    #
    #          print(" {0:8d} |".format(neighbour_ids[i,n]), end='')
    #          #  print("nb: {0:8d}  Aij: {1:14.8f} {2:14.8f} ||".format(neighbour_ids[i,n], Aijs[i,n,0], Aijs[i,n,1]), end='')
    #      print()




    dumpfile = open(python_surface_dump, 'wb')
    pickle.dump(Aijs, dumpfile)
    pickle.dump(nneighs, dumpfile)
    pickle.dump(neighbour_ids, dumpfile)
    dumpfile.close()
    print("Dumped python data")

    return





#================================
def compare_Aij():
#================================
    """
    compare the Aijs you got
    """

    # Read in data
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


    python_filep = open(python_surface_dump, 'rb')
    Aij_p = pickle.load(python_filep)
    nneigh_Aij_p = pickle.load(python_filep)
    nids_Aij_p = pickle.load(python_filep)
    python_filep.close()

    try:
        python_grad_filep = open(python_grad_dump, 'rb')
    except FileNotFoundError:
        print("File", python_grad_dump, "not found.")
        print("Did you run check_gradients.py first?")
        quit(2)
    grads_p         = pickle.load(python_grad_filep)
    sum_grad_p      = pickle.load(python_grad_filep)
    grads_contrib_p = pickle.load(python_grad_filep)
    dwdr_p          = pickle.load(python_grad_filep)
    Wjxi_p          = pickle.load(python_grad_filep)
    nids_p          = pickle.load(python_grad_filep)
    nneigh_p        = pickle.load(python_grad_filep)
    omega_p         = pickle.load(python_grad_filep)
    r_p             = pickle.load(python_grad_filep)
    dx_p            = pickle.load(python_grad_filep)
    iinds           = pickle.load(python_grad_filep)
    python_grad_filep.close()
    



    npart = nneigh_s.shape[0]
    H = ms.get_H(h)




    #---------------------------------------------
    def check_neighbours():
    #---------------------------------------------

        print("Checking number of neighbours")

        found_difference = False
        for p in range(npart):
            py = nneigh_p[p]
            sw = nneigh_s[p]
            if py != sw:
                # if a neighbour of particle i is not inside i's compact
                # support radius, but i is inside the neighbours, swift
                # will write it down anyway. So check that that isn't the case.

                diff = py - sw
                if diff > 0:
                    larger = nids_p[p, :nneigh_p[p]]
                    smaller = nids_s[p, :nneigh_s[p]]
                    text = "in pyton but not in swift"
                    found_difference = True
                    print("Difference: Python found more neighbours. ID:", ids[p])
                    print("sw:", nids_s[p, :nneigh_s[p]])
                    print("py:", nids_p[p, :nneigh_p[p]])
                else:
                    larger = nids_s[p, :nneigh_s[p]]
                    smaller = nids_p[p, :nneigh_p[p]]
                    text = "in swift but not in python"

                    swap = False
                    remove = np.zeros((nneigh_s[p]), dtype=np.int)
                    for i,n in enumerate(larger):
                        if not np.isin(n, smaller):
                            xn = pos[n-1, 0]
                            yn = pos[n-1, 1]
                            xp = pos[p, 0]
                            yp = pos[p, 1]
                            dx = ms.get_dx(xn, xp, yn, yp, L=L, periodic=periodic)
                            r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
                            if r > H[n-1]:
                                print("Neighbour", n, text)
                                print("Difference: id:", ids[p], "neighbour:", ids[n])
                                print("sw:", nids_s[p, :nneigh_s[p]])
                                print("py:", nids_p[p, :nneigh_p[p]])
                                found_difference = True
                            else:
                                # remove SWIFT extras
                                swap = True
                                remove[i] = 1

                    if swap:
                        n = 0
                        while n < nneigh_s[p]:
                            if remove[n] == 1:
                                nids_s[p,n:nneigh_s[p]-1] = nids_s[p, n+1:nneigh_s[p]]
                                Aij_s[p, 2*n:2*nneigh_s[p]-2] = Aij_s[p, 2*n+2:2*nneigh_s[p]]
                                remove[n:nneigh_s[p]-1] = remove[n+1:nneigh_s[p]]
                                nneigh_s[p] -= 1
                                n -= 1
                            n+=1


        if not found_difference:
            print("Finished, all the same.")
        else:
            print("Makes no sense to continue. Exiting")
            quit()

        return





    #------------------------------------------
    def check_neighbour_IDs():
    #------------------------------------------

        print("Checking neighbour IDs")

        found_difference = False
        for p in range(npart):
            pyarr = nids_p[p]
            swarr = nids_Aij_s[p][:nneigh_s[p]]
            add = 0
            for n in range(nneigh_p[p]):

                ns = n + add
                py = pyarr[n]
                try:
                    sw = nids_Aij_s[p][ns]
                except IndexError:
                    print("Something fucky going on")
                    print("particle:", ids[p], "neighbour swift:", sw, "neighbour py:", py, "p:", p, "n:", n)
                    print("nneigh py:", nneigh_p[p], "nneigh sw:", nneigh_Aij_s[p])
                    quit()

                while py > sw:
                    # From the SWIFT output, there may be more neighbours than from python
                    # if rij < H_j but r_ij > H_i, particle j has i as neighbour, but
                    # i hasn't got j as neighbour. Both neighbours will be written down though
                    add += 1
                    try:
                        sw = nids_Aij_s[p][n+add]
                    except IndexError:
                        print("Something fucky going on")
                        print("particle:", ids[p], "neighbour swift:", sw, "neighbour py:", py, "p:", p, "n:", n)
                        print("nneigh py:", nneigh_p[p], "nneigh sw:", nneigh_Aij_s[p])
                        quit()

                ns = n+add
                sw = nids_Aij_s[p][ns]

                if py != sw:
                    found_difference = True
                    print("Difference: id:", ids[p], "py:", py, "sw:", sw)

        if not found_difference:
            print("Finished, all the same.")
        else:
            print("Makes no sense to continue. Exiting")
            quit()

        return






    #-------------------------------
    def check_Aij():
    #-------------------------------

        print("Checking surfaces")

        found_difference = False
        for p in range(npart):

            nb = nneigh_p[p]
            maxA = Aij_p[p, :nb].max()
            maxAtot = Aij_p.max()
            null = NULL_RELATIVE*maxA

            add = 0

            for n in range(nb):

                ns = n + add
                nid_p = nids_Aij_p[p][n]
                nid_s = nids_Aij_s[p][ns]

                while nid_p > nid_s:
                    # From the SWIFT output, there may be more neighbours than from python
                    # if rij < H_j but r_ij > H_i, particle j has i as neighbour, but
                    # i hasn't got j as neighbour. Both neighbours will be written down though
                    add += 1
                    try:
                        nid_s = nids_Aij_s[p][n+add]
                    except IndexError:
                        print("Something fucky going on")
                        print("particle:", ids[p], "neighbour swift:", nid_s, "neighbour py:", nid_p, "p:", p, "n:", n)
                        print("nneigh py:", nneigh_p[p], "nneigh sw:", nneigh_Aij_s[p])
                        quit()

                ns = n+add




                nbp = nids_Aij_p[p, n]
                nbs = nids_Aij_s[p, ns]
                pyx = Aij_p[p,n,0]
                pyy = Aij_p[p,n,1]
                pyn = np.sqrt(pyx**2 + pyy**2)
                swx = Aij_s[p][2*ns]
                swy = Aij_s[p][2*ns+1]
                swn = np.sqrt(swx**2 + swy**2)

                if swn > null and pyn > null:
                    diff = 1 - pyn/swn
                    if diff > tolerance:
                        print("=========================================================================================")
                        print("Particle ID", ids[p], "neighbour id:", nbp)
                        print("Max |Aij| of this particle: {0:14.7e}, max |Aij| globally: {1:14.7e}".format(maxA, maxAtot))
                        print("lower threshold for 'zero' is: {0:14.7e}".format(null))
                        print()
                        print("              Python          Swift               |1 - py/swift|")
                        print("-----------------------------------------------------------------------------------------")
                        print("Aij x:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(pyx, swx, abs(1-pyx/swx)))
                        print("Aij y:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(pyy, swy, abs(1-pyy/swy)))
                        print("|Aij|:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(pyn, swn, abs(1-pyn/swn)))
                        print()


                        py = dwdr_p[p, n]
                        sw = dwdr_s[p, n]
                        print("dwdr:        {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = r_p[p, n]
                        sw = r_s[p, n]
                        print("r:           {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = dx_p[p, n, 0]
                        sw = dx_s[p, 2*n]
                        print("dx[0]:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = dx_p[p, n, 1]
                        sw = dx_s[p, 2*n+1]
                        print("dx[1]:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = sum_grad_p[p,0]
                        sw = sum_grad_s[p,0]
                        print("sum_grad[0]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        py = sum_grad_p[p,1]
                        sw = sum_grad_s[p,1]
                        print("sum_grad[1]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(py, sw, abs(1-py/sw)))

                        vip = 1/omega_p[p]
                        vis = vol_s[p]
                        print("volume i:    {0:14.7e}  {1:14.7e}  {2:14.7f}".format(vip, vis, abs(1-vip/vis)))

                        nind = nids_p[p,n]-1
                        vjp = 1/omega_p[nind]
                        vjs = vol_s[nind]
                        print("volume j:    {0:14.7e}  {1:14.7e}  {2:14.7f}".format(vjp, vjs, abs(1-vjp/vjs)))


                        i = iinds[p, n]
                        j = nind
                        gpix = grads_p[j, i, 0]
                        gpiy = grads_p[j, i, 1]
                        gsix = grads_s[p, 2*ns]
                        gsiy = grads_s[p, 2*ns+1]

                        print("grad fin x:  {0:14.7e}  {1:14.7e}  {2:14.7f}".format(gpix, gsix, abs(1-gpix/gsix)))
                        print("grad fin y:  {0:14.7e}  {1:14.7e}  {2:14.7f}".format(gpiy, gsiy, abs(1-gpiy/gsiy)))

                        gpjx = grads_p[p, n, 0]
                        gpjy = grads_p[p, n, 1]
                        newns = np.asscalar(np.where(nids_Aij_s[nind]==ids[p])[0])
                        gsjx = grads_s[nind, 2*newns]
                        gsjy = grads_s[nind, 2*newns+1]

                        Apx = vjp * gpix - vip * gpjx
                        Apy = vjp * gpiy - vip * gpjy
                        Asx = vjs * gsix - vis * gsjx
                        Asy = vjs * gsiy - vis * gsjy

                        print("recomp Ax:   {0:14.7e}  {1:14.7e}  {2:14.7f}".format(Apx, Asx, abs(1-Apx/Asx)))
                        print("recomp Ay:   {0:14.7e}  {1:14.7e}  {2:14.7f}".format(Apy, Asy, abs(1-Apy/Asy)))

                        #  print("Diff: {0:14.7f}".format(diff))
                        print()
                        found_difference = True

        if not found_difference:
            print("Finished, all the same.")

        return


    
    check_neighbours()
    check_neighbour_IDs()
    check_Aij()

    return







#==========================
def main():
#==========================
    
    announce()
    extract_dump_data(srcfile, swift_dump, part_dump)
    compute_Aij_my_way()
    compare_Aij()
    return




if __name__ == '__main__':

    main()

