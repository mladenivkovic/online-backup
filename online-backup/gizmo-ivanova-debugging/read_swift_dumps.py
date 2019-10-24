#!/usr/bin/env python3

# Module to read in dumps written by swift for 
# the gizmo debugging


import numpy as np
import pickle
import os
from my_utils import yesno, one_arg_present


#==========================================
def extract_dump_data(srcfile, dumpfile):
#==========================================
    """
    Reads in, sorts out and pickle dumps from swift output
    """

    if os.path.isfile(dumpfile):
        if not yesno("Dump file", dumpfile, "already exists. Shall I overwrite it?"):
            return


    #------------------
    # read in data
    #------------------

    f = open(srcfile, 'rb')
    npart = np.asscalar(np.fromfile(f, dtype=np.int64, count=1)) - 1 # -1: starting at index 1 for particle IDs
    nguess = np.asscalar(np.fromfile(f, dtype=np.int32, count=1))

    print("got npart:", npart, "nguess:", nguess)

    # set up arrays
    ids     = np.empty(npart, dtype=np.int64)
    pos     = np.empty((npart, 3), dtype=np.float32)
    h       = np.empty(npart, dtype=np.float32)
    nids    = np.empty((npart, nguess), dtype=np.int64)
    nneigh  = np.empty((npart), dtype=np.int32)
    gradsum = np.empty((npart, 3), dtype=np.float32)   # sum of all gradient contributions
    dwdr    = np.empty((npart, nguess), dtype=np.float32)
    dx      = np.empty((npart, 2*nguess), dtype=np.float32)
    vol     = np.empty((npart), dtype=np.float32)
    omega   = np.empty((npart), dtype=np.float32)
    r       = np.empty((npart, nguess), dtype=np.float32)
    grads   = np.empty((npart, 2*nguess), dtype=np.float32)          # del psi/del x 

    nneigh_Aij = np.empty((npart), dtype=np.int32)
    nids_Aij = np.empty((npart, nguess), dtype=np.int64)
    Aij    = np.empty((npart, 2*nguess), dtype=np.float32)

    for p in range(npart):
        ids[p]      = np.asscalar(np.fromfile(f, dtype=np.int64, count=1))
        h[p]        = np.asscalar(np.fromfile(f, dtype=np.float32, count=1))
        omega[p]    = np.asscalar(np.fromfile(f, dtype=np.float32, count=1))
        vol[p]      = np.asscalar(np.fromfile(f, dtype=np.float32, count=1))
        gradsum[p]  = np.fromfile(f, dtype=np.float32, count=3)
        pos[p]      = np.fromfile(f, dtype=np.float32, count=3)

        nneigh[p]   = np.asscalar(np.fromfile(f, dtype=np.int32, count=1))
        nids[p]     = np.fromfile(f, dtype=np.int64, count=nguess)
        grads[p]    = np.fromfile(f, dtype=np.float32, count=2*nguess)
        dwdr[p]     = np.fromfile(f, dtype=np.float32, count=nguess)
        dx[p]       = np.fromfile(f, dtype=np.float32, count=2*nguess)
        r[p]        = np.fromfile(f, dtype=np.float32, count=nguess)

        nneigh_Aij[p] = np.asscalar(np.fromfile(f, dtype=np.int32, count=1))
        nids_Aij[p] = np.fromfile(f, dtype=np.int64, count=nguess)
        Aij[p] = np.fromfile(f, dtype=np.float32, count=2*nguess)

        #  t = np.fromfile(f, dtype=np.int8, count=11)
        #  teststring = "".join([chr(item) for item in t])
        #  print("TESTSTRING:", teststring)






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
    dx = dx[inds]
    r = r[inds]
    nneigh_Aij = nneigh_Aij[inds]
    nids_Aij = nids_Aij[inds]
    Aij = Aij[inds]


    # sort neighbour dependent data by neighbour IDs
    for n in range(npart):
        nb = nneigh[n]
        # get indices of neighbours
        ninds = np.argsort(nids[n][:nb])
        # sort neighbour IDs
        nids[n][:nb] = nids[n][:nb][ninds]
        # sort dwdr
        dwdr[n][:nb] = dwdr[n][:nb][ninds]
        # sort r
        r[n][:nb] = r[n][:nb][ninds]
        # sort individual gradient contributions
        temp = np.empty((2*nb), dtype=np.float)
        temp_dx = np.empty((2*nb), dtype=np.float)
        for i, nn in enumerate(ninds):

            temp[2*i:2*i+2] = grads[n,2*nn:2*nn+2]
            #  temp[2*i] = Aijs[n, 2*nn]
            temp_dx[2*i:2*i+2] = dx[n,2*nn:2*nn+2]
            #  temp[2*i] = Aijs[n, 2*nn] 
        grads[n][:2*nb] = temp
        dx[n][:2*nb] = temp_dx


    for n in range(npart):
        nb = nneigh_Aij[n]
        # get indices of neighbours
        ninds = np.argsort(nids_Aij[n][:nb])
        # sort neighbour IDs
        nids_Aij[n][:nb] = nids_Aij[n][:nb][ninds]
        # sort individual gradient contributions
        temp = np.empty((2*nb), dtype=np.float)
        for i, nn in enumerate(ninds):

            temp[2*i:2*i+2] = Aij[n,2*nn:2*nn+2]

        Aij[n][:2*nb] = temp




    #------------------
    # dump
    #------------------

    data_dump = [grads, gradsum, dwdr, nids, nneigh, omega, vol, pos, h, ids, dx, r, nneigh_Aij, nids_Aij, Aij]
    dumpf= open(dumpfile, 'wb')
    pickle.dump(data_dump, dumpf)
    dumpf.close()
    print("Dumped swift data")



    return










#==========================================
def extract_gradients_from_snapshot_hdf5():
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
    dx = parts['dx'][:]
    r = parts['r'][:]

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
    dx = dx[inds]
    r = r[inds]


    # sort neighbour dependent data by neighbour IDs
    for n in range(nids.shape[0]):
        nb = nneigh[n]
        # get indices of neighbours
        ninds = np.argsort(nids[n][:nb])
        # sort neighbour IDs
        print(nids[n][:nb])
        nids[n][:nb] = nids[n][:nb][ninds]
        # sort dwdr
        print(nids[n][:nb])
        print()
        dwdr[n][:nb] = dwdr[n][:nb][ninds]
        # sort r
        r[n][:nb] = r[n][:nb][ninds]
        # sort individual gradient contributions
        temp = np.empty((2*nb), dtype=np.float)
        temp_dx = np.empty((2*nb), dtype=np.float)
        for i, nn in enumerate(ninds):

            temp[2*i:2*i+2] = grads[n,2*nn:2*nn+2]
            #  temp[2*i] = Aijs[n, 2*nn]
            temp_dx[2*i:2*i+2] = dx[n,2*nn:2*nn+2]
            #  temp[2*i] = Aijs[n, 2*nn]

        grads[n][:2*nb] = temp
        dx[n][:2*nb] = temp_dx




    #------------------
    # dump
    #------------------

    data_dump = [grads, gradsum, dwdr, nids, nneigh, omega, vol, pos, h, ids, dx, r]
    dumpfile = open(swift_dump, 'wb')
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped swift data")



    return

