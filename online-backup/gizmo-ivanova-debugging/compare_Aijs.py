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

snap = '0001'                   # which snap to use
#  hdf_prefix = 'sodShock_'        # snapshot name prefix
#  hdf_prefix = 'perturbedPlane_'  # snapshot name prefix
hdf_prefix = 'uniformPlane_'    # snapshot name prefix

srcfile = hdf_prefix+snap+'.hdf5'

swift_dump = 'dump_swift_Aij_'+snap+'.pkl'
extra_dump = 'dump_extra_particle_data_'+snap+'.pkl'
python_dump = 'dump_my_python_Aij_'+snap+'.pkl'







#==========================================
def extract_Aij_from_snapshot():
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


    Aijs = parts['Aij'][:]
    nneighs = parts['nneigh'][:] + 1 # it was used in the code as the current free index - 1, so add 1
    neighbour_ids = parts['NeighbourIDs'][:]

    f.close()



    #------------------
    # sort
    #------------------

    inds = np.argsort(ids)

    Aijs = Aijs[inds]
    nneighs = nneighs[inds]
    neighbour_ids = neighbour_ids[inds]
    for n in range(neighbour_ids.shape[0]):
        nb = nneighs[n]
        ninds = np.argsort(neighbour_ids[n][:nb])
        neighbour_ids[n][:nb] = neighbour_ids[n][:nb][ninds]
        #  neighbour_ids[n][:nb] = np.sort(neighbour_ids[n][:nb])
        temp = np.empty((2*nb), dtype=np.float)
        for i, nn in enumerate(ninds):

            temp[2*i:2*i+2] = Aijs[n,2*nn:2*nn+2]
            #  temp[2*i] = Aijs[n, 2*nn]

        Aijs[n][:2*nb] = temp


    ids = ids[inds]
    pos = pos[inds]




    #------------------
    # dump
    #------------------

    data_dump = [Aijs, nneighs, neighbour_ids]
    dumpfile = open(swift_dump, 'wb')
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped swift data")

    data_dump = [pos, ids]
    dumpfile = open(extra_dump, 'wb')
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped extra particle data")




    # note that the arrays are already sorted at this point!
    #  for i in range(5):
    #
    #      print("ID: {0:8d} {1:8d} ||".format(ids[i], nneighs[i]), end='')
    #
    #      ninds = np.argsort(neighbour_ids[i])
    #      print(ninds)
    #      print(neighbour_ids[i])
    #      for n in range(nneighs[i]):
    #
    #          # there are probably a lot of zeros coming in first, so start from behind, since you know how many there are
    #          nn = ninds[-nneighs[i]+n]
    #          print(" {0:8d} |".format(neighbour_ids[i,nn]), end='')
    #          #  print("nb: {0:8d}  Aij: {1:14.8f} {2:14.8f} ||".format(neighbour_ids[i,nn], Aijs[i,2*nn], Aijs[i,2*nn+1]), end='')
    #      print()
    #


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

    # read data from snapshot
    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, 'PartType0')
    #  x, y, h, rho, m, ids, npart = ms.read_file(srcfile, 'PartType0', sort=True)

    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)

    A_ij_all, neighbours_all = ms.Aij_Ivanova_all(x, y, H, m, rho)
   
    Aijs = np.zeros((x.shape[0], 200, 2), dtype=np.float)
    nneighs = np.zeros((x.shape[0]), dtype=np.int)
    neighbour_ids = np.zeros((x.shape[0], 200), dtype=np.int)

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
    extra_filep = open(extra_dump, 'rb')
    python_filep = open(python_dump, 'rb')

    data_swift = pickle.load(swift_filep)
    Aij_s, nneigh_s, nid_s = data_swift

    data_python = pickle.load(python_filep)
    Aij_p, nneigh_p, nid_p = data_python

    data_extra = pickle.load(extra_filep)
    pos, ids = data_extra

    swift_filep.close()
    extra_filep.close()
    python_filep.close()


    npart = nneigh_s.shape[0]




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




    print("Checking surfaces")
    found_difference = False
    #  for p in range(npart):
    for p in range(3):
        print("Particle ID", ids[p])
        nb = nneigh_p[p]
        for n in range(nb):
            nbp = nid_p[p, n]
            nbs = nid_s[p, n]
            pyx = Aij_p[p,n,0]
            pyy = Aij_p[p,n,1]
            pyn = np.sqrt(pyx**2 + pyy**2)
            swx = Aij_s[p][2*n]
            swy = Aij_s[p][2*n+1]
            swn = np.sqrt(swx**2 + swy**2)


            print("neighbour id:", nbp, nbs)
            print("Aij x:       ", pyx, swx)
            print("Aij y:       ", pyy, swy)
            print("|Aij|:       ", pyn, swn)
            print("-------------------------------------------------")
        print("========================================================================")


    return






#==========================
def main():
#==========================
    
    extract_Aij_from_snapshot()
    compute_Aij_my_way()
    #  compare_Aij()
    return




if __name__ == '__main__':

    main()

