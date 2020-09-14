#!/usr/bin/env python3


import meshless as ms
import numpy as np
import pickle
from my_utils import one_arg_present

#  srcfile = '../dummy_output-clean/dummy_0000.hdf5'    # swift output file
srcfile = "./sodShock_0002.hdf5"  # swift output file
# if cmdline arg given, use that filename
srcfile = one_arg_present(default=srcfile)
ptype = "PartType0"  # for which particle type to look for

pind = None  # index of particle you chose with pcoord
npart = 0


nbors = []  # indices of all relevant neighbour particles


def main():

    # read data from snapshot
    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype)
    #  x, y, h, rho, m, ids, npart = ms.read_file(srcfile, ptype, sort=True)

    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)

    #  A_ij_all, neighbours_all = ms.Aij_Ivanova_all(x, y, H, m, rho)
    A_ij_all, neighbours_all = ms.Aij_Ivanova_all(x, y, H)

    #  Aijs = np.zeros((x.shape[0], 200, 2), dtype=np.float)
    #  nneighs = np.zeros((x.shape[0]), dtype=np.int)
    #  neighbour_ids = np.zeros((x.shape[0], 200), dtype=np.int)
    #
    #  inds = np.argsort(ids)
    #
    #  for i, ind in enumerate(inds):
    #      # i: index in arrays to write
    #      # ind: sorted index
    #      nneighs[i] = len(neighbours_all[ind])
    #      ninds = np.argsort(np.array(neighbours_all[ind]))
    #
    #      for n in range(nneighs[i]):
    #          nind = neighbours_all[ind][ninds[n]]
    #          neighbour_ids[i, n] = ids[nind]
    #          Aijs[i, n] = A_ij_all[ind, ninds[n]]
    #
    #  data_dump = [Aijs, nneighs, neighbour_ids]
    #  dumpfile = open("dump_my_python_Aij_0002.pkl", "wb")
    #  pickle.dump(data_dump, dumpfile)
    #  dumpfile.close()
    #  print("Dumped data")


if __name__ == "__main__":
    main()
