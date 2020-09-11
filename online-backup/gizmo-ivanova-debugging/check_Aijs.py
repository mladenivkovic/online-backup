#!/usr/bin/env python3

# ========================================================
# compare results 1:1 from swift and python outputs
# intended to compare Ivanova effective surfaces
#
# CALL check_gradients.py FIRST!!!! THIS SCRIPT NEEDS
# THE OUTPUT THAT check_gradients.py PRODUCES!
# usage: ./compare_Aijs.py <dump-number>
#   dump number is optional, is binary dump that
#   I handwrote for this and implemented into swift
# ========================================================


import numpy as np
import pickle
import h5py
import os

import meshless as ms
from my_utils import yesno, one_arg_present

from filenames import get_srcfile, get_dumpfiles
from read_swift_dumps import extract_dump_data
from compute_gradients import compute_gradients_my_way

# -------------------------------
# Filenames and global stuff
# -------------------------------

periodic = True
srcfile = get_srcfile()
swift_dump, part_dump, python_surface_dump, python_grad_dump = get_dumpfiles()


# ----------------------
# Behaviour params
# ----------------------

tolerance = 5e-3  # relative tolerance threshold for relative float comparison: if (a - b)/a < tolerance, it's fine
NULL_RELATIVE = 1e-4  # relative tolerance for values to ignore below this value

do_break = True


def announce():

    print("CHECKING SURFACES.")
    print("tolerance is set to:", tolerance)
    print("NULL_RELATIVE is set to:", NULL_RELATIVE)
    print("---------------------------------------------------------")
    print()

    return


def compute_Aij_my_way():
    """
    Compute Aij using my python module, and dump results in a pickle
    """

    if os.path.isfile(python_surface_dump):
        if not yesno(
            "Dump file", python_surface_dump, "already exists. Shall I overwrite it?"
        ):
            return

    print("Computing Aij")

    part_filep = open(part_dump, "rb")
    ids = pickle.load(part_filep)
    pos = pickle.load(part_filep)
    h = pickle.load(part_filep)
    part_filep.close()
    H = ms.get_H(h)

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
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
        nbs = np.array(neighbours_all[ind])  # list of neighbours of current particle
        nneighs[i] = nbs.shape[0]
        ninds = np.argsort(ids[nbs])  # indices of neighbours in nbs array sorted by IDs
        #  ninds = np.argsort(np.array(ids[nbs]))  # indices of neighbours in nbs array sorted by IDs

        for n in range(nneighs[i]):
            nind = nbs[
                ninds[n]
            ]  # index of n-th neighbour to write in increasing ID order in global arrays
            neighbour_ids[i, n] = ids[nind]
            Aijs[i, n] = A_ij_all[ind, ninds[n]]

    dumpfile = open(python_surface_dump, "wb")
    pickle.dump(Aijs, dumpfile)
    pickle.dump(nneighs, dumpfile)
    pickle.dump(neighbour_ids, dumpfile)
    dumpfile.close()
    print("Dumped python data")

    return


def compare_Aij():
    """
    compare the Aijs you got
    """

    # Read in data
    swift_filep = open(swift_dump, "rb")
    grads_s = pickle.load(swift_filep)
    grads_contrib_s = pickle.load(swift_filep)
    sum_grad_s = pickle.load(swift_filep)
    dwdr_s = pickle.load(swift_filep)
    Wjxi_s = pickle.load(swift_filep)
    nids_s = pickle.load(swift_filep)
    nneigh_s = pickle.load(swift_filep)
    omega_s = pickle.load(swift_filep)
    vol_s = pickle.load(swift_filep)
    dx_s = pickle.load(swift_filep)
    r_s = pickle.load(swift_filep)
    nneigh_Aij_s = pickle.load(swift_filep)
    nids_Aij_s = pickle.load(swift_filep)
    Aij_s = pickle.load(swift_filep)
    swift_filep.close()

    part_filep = open(part_dump, "rb")
    ids = pickle.load(part_filep)
    pos = pickle.load(part_filep)
    h = pickle.load(part_filep)
    part_filep.close()
    H = ms.get_H(h)

    python_filep = open(python_surface_dump, "rb")
    Aij_p = pickle.load(python_filep)
    nneigh_Aij_p = pickle.load(python_filep)
    nids_Aij_p = pickle.load(python_filep)
    python_filep.close()

    try:
        python_grad_filep = open(python_grad_dump, "rb")
    except FileNotFoundError:
        print("File", python_grad_dump, "not found.")
        print("Did you run check_gradients.py first?")
        quit(2)
    grads_p = pickle.load(python_grad_filep)
    sum_grad_p = pickle.load(python_grad_filep)
    grads_contrib_p = pickle.load(python_grad_filep)
    dwdr_p = pickle.load(python_grad_filep)
    Wjxi_p = pickle.load(python_grad_filep)
    nids_p = pickle.load(python_grad_filep)
    nneigh_p = pickle.load(python_grad_filep)
    omega_p = pickle.load(python_grad_filep)
    r_p = pickle.load(python_grad_filep)
    dx_p = pickle.load(python_grad_filep)
    iinds = pickle.load(python_grad_filep)
    python_grad_filep.close()

    npart = nneigh_s.shape[0]
    H = ms.get_H(h)
    L = ms.read_boxsize()

    # -----------------------------------------------
    def break_now(nis, nip, p, for_Aij=False):
        # -----------------------------------------------
        if for_Aij:
            if nis >= nneigh_Aij_s[p]:
                return True
        else:
            if nis >= nneigh_s[p]:
                return True
        if nip >= nneigh_p[p]:
            return True
        return False

    found_difference = False
    maxAtot = np.absolute(Aij_p).max()

    for p in range(npart):

        nb = nneigh_p[p]
        maxA = np.absolute(Aij_p[p, :nb]).max()
        maxAx = np.absolute(Aij_p[p, :nb, 0]).max()
        maxAy = np.absolute(Aij_p[p, :nb, 1]).max()
        null = NULL_RELATIVE * maxA
        nullx = NULL_RELATIVE * maxAx
        nully = NULL_RELATIVE * maxAy

        nis = 0
        nip = 0

        not_checked_py = [True for i in range(nneigh_p[p])]
        not_checked_sw = [True for i in range(nneigh_Aij_s[p])]

        while True:
            if break_now(nis, nip, p, for_Aij=True):
                break
            while nids_Aij_s[p, nis] != nids_Aij_p[p, nip]:
                if nids_Aij_s[p, nis] < nids_Aij_p[p, nip]:
                    nis += 1
                    continue
                elif nids_Aij_s[p, nis] > nids_Aij_p[p, nip]:
                    nip += 1
                    continue
                if break_now(nis, nip, p, for_Aij=True):
                    break

            not_checked_py[nip] = False
            not_checked_sw[nis] = False

            nind = nids_Aij_p[p, nip] - 1

            nbp = nids_Aij_p[p, nip]
            nbs = nids_Aij_s[p, nis]
            pyx = Aij_p[p, nip, 0]
            pyy = Aij_p[p, nip, 1]
            pyn = np.sqrt(pyx ** 2 + pyy ** 2)
            swx = Aij_s[p][2 * nis]
            swy = Aij_s[p][2 * nis + 1]
            swn = np.sqrt(swx ** 2 + swy ** 2)

            #  for P, S, N in [(pyx, swx, nullx), (pyy, swy, nully), (pyn, swn, null)]:
            for P, S, N in [(pyn, swn, null)]:

                if P > N and S > N:
                    diff = abs(1 - abs(P / S))

                    if diff > tolerance:
                        print(
                            "========================================================================================="
                        )
                        print("Particle ID", ids[p], "neighbour id:", nbp)
                        print(
                            "Max |Aij| of this particle: {0:14.7e}, max |Aij| globally: {1:14.7e}".format(
                                maxA, maxAtot
                            )
                        )
                        print("lower threshold for 'zero' is: {0:14.7e}".format(null))
                        print()
                        print(
                            "              Python          Swift               |1 - py/swift|"
                        )
                        print(
                            "-----------------------------------------------------------------------------------------"
                        )
                        print(
                            "Aij x:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                pyx, swx, abs(1 - pyx / swx)
                            )
                        )
                        print(
                            "Aij y:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                pyy, swy, abs(1 - pyy / swy)
                            )
                        )
                        print(
                            "|Aij|:       {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                pyn, swn, abs(1 - pyn / swn)
                            )
                        )
                        print()

                        NIS = np.asscalar(
                            np.where(nids_s[p, : nneigh_s[p]] == nids_Aij_s[p, nis])[0]
                        )
                        if NIS != nis:
                            print(
                                "==================================== NIS:",
                                NIS,
                                "nis:",
                                nis,
                            )
                            print(nids_Aij_s[p, : nneigh_Aij_s[p]])
                            print(nids_s[p, : nneigh_s[p]])

                        py = Wjxi_p[p, nip]
                        sw = Wjxi_s[p, NIS]
                        rpwjxi = py
                        rswjxi = sw
                        print(
                            "              Wj(xi): {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                py, sw, abs(1 - py / sw)
                            )
                        )

                        py = dwdr_p[p, nip]
                        sw = dwdr_s[p, NIS]
                        rpdwdr = py
                        rsdwdr = sw
                        print(
                            "                dwdr: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                py, sw, abs(1 - py / sw)
                            )
                        )

                        py = r_p[p, nip]
                        sw = r_s[p, NIS]
                        rpr = py
                        rsr = sw
                        print(
                            "                   r: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                py, sw, abs(1 - py / sw)
                            )
                        )

                        py = dx_p[p, nip, 0]
                        sw = dx_s[p, 2 * NIS]
                        rpdx = py
                        rsdx = sw
                        print(
                            "               dx[0]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                py, sw, abs(1 - py / sw)
                            )
                        )

                        py = dx_p[p, nip, 1]
                        sw = dx_s[p, 2 * NIS + 1]
                        rpdy = py
                        rsdy = sw
                        print(
                            "               dx[1]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                py, sw, abs(1 - py / sw)
                            )
                        )

                        py = sum_grad_p[p, 0]
                        sw = sum_grad_s[p, 0]
                        rpsumx = py
                        rssumx = sw
                        print(
                            "         sum_grad[0]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                py, sw, abs(1 - py / sw)
                            )
                        )

                        py = sum_grad_p[p, 1]
                        sw = sum_grad_s[p, 1]
                        rpsumy = py
                        rssumy = sw
                        print(
                            "         sum_grad[1]: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                py, sw, abs(1 - py / sw)
                            )
                        )

                        py = omega_p[p]
                        sw = omega_s[p]
                        rpom = py
                        rsom = sw
                        print(
                            "             1/omega: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                1 / py, 1 / sw, abs(1 - sw / py)
                            )
                        )

                        print()

                        rpdwdx = rpdwdr * rpdx / rpr
                        rpdwdy = rpdwdr * rpdy / rpr
                        rsdwdx = rsdwdr * rsdx / rsr
                        rsdwdy = rsdwdr * rsdy / rsr

                        rpx = rpdwdx / rpom - rpwjxi * rpsumx / rpom ** 2
                        rpy = rpdwdy / rpom - rpwjxi * rpsumy / rpom ** 2
                        rsx = rsdwdx / rsom - rswjxi * rssumx / rsom ** 2
                        rsy = rsdwdy / rsom - rswjxi * rssumy / rsom ** 2

                        iind = iinds[p, nip]
                        gpix = grads_p[p, nip, 0]
                        gpiy = grads_p[p, nip, 1]
                        gsix = grads_s[p, 2 * nis]
                        gsiy = grads_s[p, 2 * nis + 1]

                        gpjx = grads_p[nind, iind, 0]
                        gpjy = grads_p[nind, iind, 1]
                        newns = np.asscalar(np.where(nids_Aij_s[nind] == ids[p])[0])
                        gsjx = grads_s[nind, 2 * newns]
                        gsjy = grads_s[nind, 2 * newns + 1]

                        vjp = 1 / omega_p[nind]
                        vip = 1 / omega_p[p]
                        vis = vol_s[p]
                        vjs = vol_s[nind]

                        Apx = vip * gpix - vjp * gpjx
                        Apy = vip * gpiy - vjp * gpjy
                        Asx = vis * gsix - vjs * gsjx
                        Asy = vis * gsiy - vjs * gsjy

                        print(vip, vis)
                        print(vjp, vjs)

                        print(" recomputed :")
                        print(
                            "             dW / dx: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                rpdwdx, rsdwdx, abs(1 - rpdwdx / rsdwdx)
                            )
                        )
                        print(
                            "             dW / dy: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                rpdwdy, rsdwdy, abs(1 - rpdwdy / rsdwdy)
                            )
                        )
                        print(
                            "     del psi / del x: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                rpx, rsx, abs(1 - rpx / rsx)
                            )
                        )
                        print(
                            "     del psi / del y: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                rpy, rsy, abs(1 - rpy / rsy)
                            )
                        )

                        print(
                            "                  Ax: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                Apx, Asx, abs(1 - Apx / Asx)
                            )
                        )
                        print(
                            "                  Ay: {0:14.7e}  {1:14.7e}  {2:14.7f}".format(
                                Apy, Asy, abs(1 - Apy / Asy)
                            )
                        )

                        #  print("Diff: {0:14.7f}".format(diff))
                        print()
                        found_difference = True

                    if found_difference and do_break:  # for P, S, N in ...
                        break

            if found_difference and do_break:  # neighbour loop
                break
            nis += 1
            nip += 1

        if found_difference and do_break:  # particle loop
            break
        else:
            for n, not_checked in enumerate(not_checked_py):
                if not_checked:
                    nind = nids_p[p, n] - 1
                    dx, dy = ms.get_dx(
                        pos[p, 0],
                        pos[nind, 0],
                        pos[p, 1],
                        pos[nind, 1],
                        L=L,
                        periodic=periodic,
                    )
                    r = np.sqrt(dx ** 2 + dy ** 2)
                    print(
                        "Not checked in python array: id {0:6d}, neighbour {1:6d}, r/H[i]: {2:14.7f} r/H[j]: {3:14.7f}".format(
                            ids[p], ids[nind], r / H[p], r / H[nind]
                        )
                    )
            for n, not_checked in enumerate(not_checked_sw):
                if not_checked:
                    nind = nids_Aij_s[p, n] - 1
                    dx, dy = ms.get_dx(
                        pos[p, 0],
                        pos[nind, 0],
                        pos[p, 1],
                        pos[nind, 1],
                        L=L,
                        periodic=periodic,
                    )
                    r = np.sqrt(dx ** 2 + dy ** 2)
                    print(
                        "Not checked in swift array: id {0:6d}, neighbour {1:6d}, r/H[i]: {2:14.7f} r/H[j]: {3:14.7f}".format(
                            ids[p], ids[nind], r / H[p], r / H[nind]
                        )
                    )

    if not found_difference:
        print("Finished, all the same.")

    return

    check_neighbours()
    check_neighbour_IDs()
    check_Aij()

    return


def main():

    announce()
    extract_dump_data()
    compute_gradients_my_way(periodic)
    compute_Aij_my_way()
    compare_Aij()
    return


if __name__ == "__main__":

    main()
