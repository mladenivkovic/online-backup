#!/usr/bin/env python3

# ========================================================
# intended to compare Ivanova effective surfaces
# you should have written them in hdf5 files from SWIFT
# as extra debug output
# currently hardcoded: up to 200 neighbours
# just compare first neighbour values
# ========================================================


import numpy as np
import pickle
import h5py
import os
import meshless as ms
from my_utils import yesno

# ------------------------
# Filenames
# ------------------------

snap = "0002"  # which snap to use
#  hdf_prefix = 'sodShock_'        # snapshot name prefix
#  hdf_prefix = 'perturbedPlane_'  # snapshot name prefix
hdf_prefix = "uniformPlane_"  # snapshot name prefix

srcfile = hdf_prefix + snap + ".hdf5"


def extract_Aij_from_snapshot():
    """
    Reads in, sorts by IDs from swift output
    """

    # ------------------
    # read in data
    # ------------------

    f = h5py.File(srcfile, "r")
    parts = f["PartType0"]
    ids = parts["ParticleIDs"][:]
    pos = parts["Coordinates"][:]

    Aijs = parts["Aij"][:]
    nneighs = (
        parts["nneigh"][:] + 1
    )  # it was used in the code as the current free index - 1, so add 1
    neighbour_ids = parts["NeighbourIDs"][:]

    f.close()

    # ------------------
    # sort
    # ------------------

    inds = np.argsort(ids)

    Aijs = Aijs[inds]
    nneighs = nneighs[inds]
    neighbour_ids = neighbour_ids[inds]

    ids = ids[inds]
    pos = pos[inds]

    return Aijs, nneighs, neighbour_ids, ids, pos


def compute_Aij_my_way():
    """
    Compute Aij using my python module, and dump results in a pickle
    """

    # read data from snapshot
    x, y, h, rho, m, ids, npart = ms.read_file(srcfile, "PartType0")
    #  x, y, h, rho, m, ids, npart = ms.read_file(srcfile, 'PartType0', sort=True)

    # get kernel support radius instead of smoothing length
    H = ms.get_H(h)

    A_ij_all, neighbours_all = ms.Aij_Ivanova_all(x, y, H, m, rho)

    inds = np.argsort(ids)

    Aij_p = A_ij_all[inds]
    nneighs_p = np.array([len(n) for n in neighbours_all], dtype=np.int)[inds]
    maxlen = np.max(nneighs_p)
    nids_p = np.zeros((nneighs_p.shape[0], maxlen), dtype=np.int)
    for i, n in enumerate(neighbours_all):
        l = len(n)
        nids_p[i][:l] = ids[n]
    nids_p = nids_p[inds]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return Aij_p, nneighs_p, nids_p


def compare_Aij(Aij_s, nneigh_s, nids_s, Aij_p, nneigh_p, nids_p, ids):
    """
    compare the Aijs you got
    """

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

    print("Checking surfaces of first neighbour")
    found_difference = False
    for p in range(npart):
        #  for p in range(10):
        print("Particle ID", ids[p])

        nb = nneigh_p[p]
        for n in range(nb):
            #  for n in range(1):
            nbp = nids_p[p, n]
            pyx = Aij_p[p, n, 0]
            pyy = Aij_p[p, n, 1]
            pyn = np.sqrt(pyx ** 2 + pyy ** 2)

            ns = np.where(nids_s[p] == nbp)
            try:
                ns = np.asscalar(ns[0])
            except ValueError:
                print(
                    "Didn't find neighbour",
                    nbp,
                    "in swift array for particle ID",
                    ids[p],
                )
                print(nids_s[p])
                quit()
            print("index in swift array:", ns)

            nbs = nids_s[p, ns]
            swx = Aij_s[p][2 * ns]
            swy = Aij_s[p][2 * ns + 1]
            swn = np.sqrt(swx ** 2 + swy ** 2)

            print("neighbour id:", nbp, nbs)
            print("Aij x:       ", pyx, swx)
            print("Aij y:       ", pyy, swy)
            print("|Aij|:       ", pyn, swn)
            print("-------------------------------------------------")
        print(
            "========================================================================"
        )

    return


def main():

    Aij_s, nneigh_s, nids_s, ids, pos = extract_Aij_from_snapshot()
    Aij_p, nneigh_p, nids_p = compute_Aij_my_way()
    compare_Aij(Aij_s, nneigh_s, nids_s, Aij_p, nneigh_p, nids_p, ids)
    return


if __name__ == "__main__":

    main()
