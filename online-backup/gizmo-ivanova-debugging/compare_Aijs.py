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
from my_utils import yesno, one_arg_present
from read_swift_dumps import extract_dump_data

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
python_dump = 'gizmo-debug-python-surface-data_'+snap+'.pkl'


#----------------------
# Behaviour params
#----------------------

tolerance = 1e-3    # relative tolerance threshold for relative float comparison: if (a - b)/a < tolerance, it's fine
NULL = 1e-3         # max(Aij) of this particle * NULL = lower limit for values to be treated as zero






#=====================
def announce():
#=====================

    print("CHECKING SURFACES.")
    print("tolerance is set to:", tolerance)
    print("NULL is set to:", NULL)
    print("---------------------------------------------------------")
    print()

    return











#========================================
def compute_Aij_my_way():
#========================================
    """
    Compute Aij using my python module, and dump results in a pickle
    """


    if os.path.isfile(python_dump):
        if not yesno("Dump file", python_dump, "already exists. Shall I overwrite it?"):
            return


    print("Computing Aij")

    part_filep = open(part_dump, 'rb')
    data_part = pickle.load(part_filep)
    ids, pos, h = data_part

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    npart = x.shape[0]

    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)
    m = 1
    rho = 1

    A_ij_all, neighbours_all = ms.Aij_Ivanova_all(x, y, H, m, rho)
   
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




    data_dump = [Aijs, nneighs, neighbour_ids]
    dumpfile = open(python_dump, 'wb')
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped python data")

    return





#================================
def compare_Aij():
#================================
    """
    compare the Aijs you got
    """


    swift_filep = open(swift_dump, 'rb')
    python_filep = open(python_dump, 'rb')
    part_filep = open(part_dump, 'rb')

    data_swift = pickle.load(swift_filep)
    grads_s, sum_grad_s, dwdr_s, nid_s, nneigh_s, omega_s, vol_s, dx_s, r_s, nneigh_Aij_s, nids_Aij_s, Aij_s = data_swift

    python_filep = open(python_dump, 'rb')
    data_python = pickle.load(python_filep)
    Aij_p, nneigh_p, nid_p = data_python

    data_part = pickle.load(part_filep)
    ids, pos, h = data_part
    H = ms.get_H(h)

    swift_filep.close()
    python_filep.close()
    part_filep.close()


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
                    larger = nid_p[p, :nneigh_p[p]]
                    smaller = nid_s[p, :nneigh_s[p]]
                    text = "in pyton but not in swift"
                    found_difference = True
                    print("Difference: Python found more neighbours. ID:", ids[p])
                    print("sw:", nid_s[p, :nneigh_s[p]])
                    print("py:", nid_p[p, :nneigh_p[p]])
                else:
                    larger = nid_s[p, :nneigh_s[p]]
                    smaller = nid_p[p, :nneigh_p[p]]
                    text = "in swift but not in python"

                    swap = False
                    remove = np.zeros((nneigh_s[p]), dtype=np.int)
                    for i,n in enumerate(larger):
                        if not np.isin(n, smaller):
                            xn = pos[n-1, 0]
                            yn = pos[n-1, 1]
                            xp = pos[p, 0]
                            yp = pos[p, 1]
                            dx = ms.get_dx(xn, xp, yn, yp)
                            r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
                            if r > H[n-1]:
                                print("Neighbour", n, text)
                                print("Difference: id:", ids[p], "neighbour:", ids[n])
                                print("sw:", nid_s[p, :nneigh_s[p]])
                                print("py:", nid_p[p, :nneigh_p[p]])
                                found_difference = True
                            else:
                                # remove SWIFT extras
                                swap = True
                                remove[i] = 1

                    if swap:
                        n = 0
                        while n < nneigh_s[p]:
                            if remove[n] == 1:
                                nid_s[p,n:nneigh_s[p]-1] = nid_s[p, n+1:nneigh_s[p]]
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
            pyarr = nid_p[p]
            swarr = nid_s[p][:nneigh_s[p]]
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

        return






    #-------------------------------
    def check_Aij():
    #-------------------------------

        print("Checking surfaces")

        found_difference = False
        for p in range(npart):
            nb = nneigh_p[p]
            maxA = Aij_p[p, :nb].max()
            null = NULL*maxA
            for n in range(nb):
                nbp = nid_p[p, n]
                nbs = nid_s[p, n]
                pyx = Aij_p[p,n,0]
                pyy = Aij_p[p,n,1]
                pyn = np.sqrt(pyx**2 + pyy**2)
                swx = Aij_s[p][2*n]
                swy = Aij_s[p][2*n+1]
                swn = np.sqrt(swx**2 + swy**2)

                if swn > null and pyn > null:
                    diff = 1 - pyn/swn
                    if diff > tolerance:
                        print("Particle ID", ids[p], "neighbour id:", nbp)
                        print("              Python         Swift          py/swift")
                        print("Aij x:       {0:14.7e} {1:14.7e} {2:14.7f}".format(pyx, swx, pyx/swx))
                        print("Aij y:       {0:14.7e} {1:14.7e} {2:14.7f}".format(pyy, swy, pyy/swy))
                        print("|Aij|:       {0:14.7e} {1:14.7e} {2:14.7f}".format(pyn, swn, pyn/swn))
                        print("diff:", diff, "; lower threshold for zero is {0:14.7e}".format(null))
                        print('{0:14.7e}'.format(maxA))
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

