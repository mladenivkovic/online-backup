#!/usr/bin/env python3


import meshless as ms
import numpy as np
import pickle

id_of_part = 1  # for which particle to work for
#  srcfile = '../dummy_output-clean/dummy_0000.hdf5'    # swift output file
srcfile = "./sodShock_0002.hdf5"  # swift output file
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

    # find index in the read in arrays that we want to work with
    #  pind = np.asscalar(np.where(ids==1)[0])
    #  pind = ms.find_index_by_id(ids, id_of_part)
    #  print("pind=", pind, 'coordinates are ', x[pind], y[pind])
    #  # find that particle's neighbours
    #  nbors = ms.find_neighbours(pind, x, y, H)

    A_ij_all, neighbours_all = ms.Aij_Ivanova_all(x, y, H, m, rho)
    #  A_ij = ms.Aij_Hopkins(pind, x, y, H, m, rho)
    #  x_ij = ms.x_ij(pind, x, y, H, nbors=nbors)

    #  for i in range(A_ij.shape[0]):
    #      n = nbors[i]
    #      dx, dy = ms.get_dx(x[pind], x[n], y[pind], y[n]);
    #      dist = np.sqrt(dx**2+dy**2)
    #      print("ID_j {0:5d} | Aij_x {1:10.4f}  Aij_y {2:10.4f} | xj {3:10.4f} yj {4:10.4f} | hj {5:10.4f} hi {6:10.4f} | dist_ij {7:10.4f}  max_dist {8:10.4f}".format(
    #                      ids[n], A_ij[i][0], A_ij[i][1], x[n], y[n], h[n], h[pind], dist, 1.778002*h[i]
    #                  )
    #              )

    Aijs = np.zeros((x.shape[0], 200, 2), dtype=np.float)
    nneighs = np.zeros((x.shape[0]), dtype=np.int)
    neighbour_ids = np.zeros((x.shape[0], 200), dtype=np.int)

    inds = np.argsort(ids)

    for i, ind in enumerate(inds):
        # i: index in arrays to write
        # ind: sorted index
        nneighs[i] = len(neighbours_all[ind])
        ninds = np.argsort(np.array(neighbours_all[ind]))

        for n in range(nneighs[i]):
            nind = neighbours_all[ind][ninds[n]]
            neighbour_ids[i, n] = ids[nind]
            Aijs[i, n] = A_ij_all[ind, ninds[n]]

    #  for i in range(10):
    #
    #      print("ID: {0:8d} ".format(ids[inds[i]]), end='')
    #
    #      for n in range(nneighs[i]):
    #
    #          print("nb: {0:8d}  Aij: {1:14.8f} {2:14.8f} ||".format(neighbour_ids[i,n], Aijs[i,n,0], Aijs[i,n,1]), end='')
    #      print()

    data_dump = [Aijs, nneighs, neighbour_ids]
    dumpfile = open("dump_my_python_Aij_0002.pkl", "wb")
    pickle.dump(data_dump, dumpfile)
    dumpfile.close()
    print("Dumped data")


if __name__ == "__main__":
    main()
