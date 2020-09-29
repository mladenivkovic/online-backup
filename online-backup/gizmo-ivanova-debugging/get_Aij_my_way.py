#!/usr/bin/env python3


import astro_meshless_surfaces as ml
import numpy as np
import pickle
from my_utils import one_arg_present

#  srcfile = '../dummy_output-clean/dummy_0000.hdf5'    # swift output file
srcfile = "./sodShock_0002.hdf5"  # swift output file

# if cmdline arg given, use that filename
srcfile = one_arg_present(default=srcfile)
ptype = "PartType0"  # for which particle type to look for


# get snap number string
# assumes it ends with XXXX.hdf5
nrstring = srcfile[-9:-5]
print("NRSTRING", nrstring)


def main():

    # read data from snapshot
    print("reading file")
    x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype)
    #  x, y, h, rho, m, ids, npart = ml.read_file(srcfile, ptype, sort=True)

    # get kernel support radius instead of smoothing length
    H = ml.get_H(h)

    #  A_ij_all, neighbours_all = ml.Aij_Ivanova_all(x, y, H, m, rho)
    print("Computing surfaces")
    A_ij_all, neighbours_all, nneigh = ml.Aij_Ivanova_all(x, y, H)

    print("Postprocessing")
    Aijs = np.zeros((x.shape[0], 200, 2), dtype=np.float)
    nneighs = np.zeros((x.shape[0]), dtype=np.int)
    neighbour_ids = np.zeros((x.shape[0], 200), dtype=np.int)

    inds = np.argsort(ids)

    for i, ind in enumerate(inds):
        # i: index in arrays to write
        # ind: sorted index
        ninds = np.argsort(np.array(neighbours_all[ind]))

        for n in range(nneighs[ind]):
            nind = neighbours_all[ind][ninds[n]]
            neighbour_ids[i, n] = ids[nind]
            Aijs[i, n] = A_ij_all[ind, ninds[n]]

    data_dump = [Aijs, nneighs, neighbour_ids]
    dumpfile = open("dump_my_python_Aij_" + nrstring + ".pkl", "wb")
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped data")


if __name__ == "__main__":
    main()
