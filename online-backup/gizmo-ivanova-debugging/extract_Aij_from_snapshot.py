#!/usr/bin/env python3


# extract, sort and print out nicely effective surfaces written in
# swift hdf5 outputs for debugging purposes


snap = 'sodShock_0002.hdf5'

import h5py
import numpy as np
import pickle


f = h5py.File(snap, 'r')
parts = f['PartType0']
ids = parts['ParticleIDs'][:]
pos = parts['Coordinates'][:]


Aijs = parts['Aij'][:]
nneighs = parts['nneigh'][:] + 1 # it was used in the code as the current free index - 1, so add 1
neighbour_ids = parts['NeighbourIDs'][:]



inds = np.argsort(ids)

Aijs = Aijs[inds]
nneighs = nneighs[inds]
neighbour_ids = neighbour_ids[inds]

ids = ids[inds]
pos = pos[inds]


data_dump = [Aijs, nneighs, neighbour_ids]
dumpfile = open('dump_swift_Aij_0002.pkl', 'wb')
pickle.dump(data_dump, dumpfile)
dumpfile.close()
print("Dumped data")

data_dump = [pos, ids]
dumpfile = open('dump_extra_particle_data_0002.pkl', 'wb')
pickle.dump(data_dump, dumpfile)
dumpfile.close()
print("Dumped extra data")




#  for i in inds:
for i in inds[0:30]:

    print("ID: {0:8d} ".format(ids[i]), end='')
    print(nneighs[i])

    ninds = np.argsort(neighbour_ids[i])
    #  print(ninds)
    #  print(neighbour_ids[i])
    for n in range(nneighs[i]):

        # there are probably a lot of zeros coming in first, so start from behind, since you know how many there are
        nn = ninds[-nneighs[i]+n]
        print("nb: {0:8d}  Aij: {1:14.8f} {2:14.8f} ||".format(neighbour_ids[i,nn], Aijs[i,2*nn], Aijs[i,2*nn+1]), end='')
    print()




