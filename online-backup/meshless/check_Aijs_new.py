#!/usr/bin/env python3

# ========================================================
# compare results 1:1 from swift and python outputs
# intended to compare Ivanova effective surfaces
#
# usage: ./compare_Aijs_new.py <dump-number>
# ========================================================


import numpy as np
import pickle
import h5py
import os
import sys

import astro_meshless_surfaces as ml


# -------------------------------
# Filenames and global stuff
# -------------------------------

periodic = True
kernel = 'cubic_spline'


# ----------------------
# Behaviour params
# ----------------------

tolerance = 1e-6  # relative tolerance threshold for relative float comparison: if (a - b)/a < tolerance, it's fine
zero_tolerance = 0. # below which value to treat float sums as zero
#  zero_tolerance = 1e-6 # below which value to treat float sums as zero
NULL_RELATIVE = 1e-4  # relative tolerance for values to ignore below this value

do_break = True
do_break_on_first = False



def get_snapfile():
    """
    Get the snapshot from given cmdline dump number
    """

    try:
        snapnr = int(sys.argv[1])
    except IndexError:
        print("You need to give me the snapshot number to work with")
        quit()

    dirlist = os.listdir()
    h5files = []
    for f in dirlist:
        if f.endswith(".hdf5"):
            snap = f[-9:-5]
            skip = False
            for letter in snap:
                if letter not in '0123456789':
                    skip = True
                    break
            if skip:
                continue
            else:
                h5files.append(f)

    h5files = sorted(h5files)


    snapfile = None
    for f in h5files:
        base = f[:-10]
        snap = f[-9:-5]
        if snap == '{0:04d}'.format(snapnr):
            snapfile = f

    if snapfile is None:
        wishfile = base + "_{0:04d}.hdf5".format(snapnr)
        raise ValueError("Couldn't find file", wishfile)

    return snapfile




def read_swift_Aij_data(snapfile):
    '''
    Reads in the Aij related data in the snapshot
    '''

    f = h5py.File(snapfile, "r")
    gas = f["PartType0"]
    ids = gas["ParticleIDs"][()]
    sortind = ids.argsort()
    Aij_x = np.squeeze(gas["Aijx"][:][sortind])
    Aij_y = np.squeeze(gas["Aijy"][:][sortind])
    Aij_z = np.squeeze(gas["Aijz"][:][sortind])
    grad_x = np.squeeze(gas["gradX"][:][sortind])
    grad_y = np.squeeze(gas["gradY"][:][sortind])
    grad_z = np.squeeze(gas["gradZ"][:][sortind])
    neighs = np.squeeze(gas["neighs"][:][sortind])
    nneighs = np.squeeze(gas["nneighs"][:][sortind])

    for i, n in enumerate(nneighs):
        ninds = neighs[i, :n].argsort()
        Aij_x[i,:n] = Aij_x[i,ninds]
        Aij_y[i,:n] = Aij_y[i,ninds]
        Aij_z[i,:n] = Aij_z[i,ninds]
        grad_x[i,:n] = grad_x[i,ninds]
        grad_y[i,:n] = grad_y[i,ninds]
        grad_z[i,:n] = grad_z[i,ninds]
        neighs[i,:n] = neighs[i, ninds]

    Aij = np.dstack((Aij_x, Aij_y, Aij_z))
    grad = np.dstack((grad_x, grad_y, grad_z))

    f.close()

    return Aij, grad, neighs, nneighs




def read_swift_particle_data(snapfile):
    '''
    Reads in the Aij related data in the snapshot
    '''

    f = h5py.File(snapfile, "r")
    gas = f["PartType0"]
    ids = gas["ParticleIDs"][:]
    sortind = ids.argsort()
    ids = ids[sortind]
    pos = np.squeeze(gas["Coordinates"][:][sortind])
    rho = np.squeeze(gas["Densities"][:][sortind])
    m = np.squeeze(gas["Masses"][:][sortind])
    h = np.squeeze(gas["SmoothingLengths"][:][sortind])

    return  pos[:,0], pos[:,1], rho, m, h, ids




def check_neighbours(neigh_s, nneigh_s, neigh_p, nneigh_p, x, y, h, L):
    """
    Check that you found the same neighbours
    neigh_s, neigh_p are paricle IDs, not indexes in arrays
    """

    print("checking neighbour counts")
    if (nneigh_s != nneigh_p ).any():
        # we got differences :(
        # go one by one
        for i in range(nneigh_s.shape[0]):
            if nneigh_s[i] != nneigh_p[i]:

                setp = set(neigh_p[i, :nneigh_p[i]])
                sets = set(neigh_s[i, :nneigh_s[i]])
                p_not_in_s = setp.difference(sets)
                s_not_in_p = sets.difference(setp)

                print("particle ", i+1)
                print("-- py", setp)
                print("-- sw", sets)
                print("-- py not in sw", p_not_in_s)
                print("-- sw not in py", s_not_in_p)
                for p in p_not_in_s:
                    j = p - 1
                    dx = ml.get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)
                    r2 = dx[0]**2 + dx[1]**2
                    Hi = ml.get_H(h[i], kernel=kernel)
                    Hj = ml.get_H(h[j], kernel=kernel)

                    line = "---   py {0:5d} {1:.3f} {2:.3f}".format(p, np.sqrt(r2/Hi**2), np.sqrt(r2/Hj**2))
                    print(line)

                for s in s_not_in_p:
                    j = s - 1
                    dx = ml.get_dx(x[i], x[j], y[i], y[j], L=L, periodic=periodic)
                    r2 = dx[0]**2 + dx[1]**2
                    Hi = ml.get_H(h[i], kernel=kernel)
                    Hj = ml.get_H(h[j], kernel=kernel)

                    line = "--- sw   {0:5d} {1:.3f} {2:.3f}".format(s, np.sqrt(r2/Hi**2), np.sqrt(r2/Hj**2))
                    print(line)

                if do_break_on_first:
                    break
            if do_break:
                break
    print("finished.")


    print("Checking Neighbour IDs")
    
    for i in range(nneigh_s.shape[0]):
        ns = neigh_s[i, :nneigh_s[i]]
        ns.sort()
        np = neigh_p[i, :nneigh_p[i]]
        np.sort()

        if (ns != np).any():
            # oh no :(
            print("oh no :(")

            if do_break:
                break

    print("finished.")





def compare_Aijs(Aij_s, neigh_s, nneigh_s, Aij_p, neigh_p, nneigh_p):
    """
    Compare Aij results from python and swift
    """


    normAij_s = Aij_s[:,:, 0]**2 + Aij_s[:,:,1]**2
    normAij_p = Aij_p[:,:, 0]**2 + Aij_p[:,:,1]**2

    print("Checking Aij norms")
    for i in range(nneigh_s.shape[0]):
        ninds = neigh_s[i, :nneigh_s[i]].argsort()
        nindp = neigh_p[i, :nneigh_p[i]].argsort()
        
        for j in range(nneigh_s[i]):
            if abs(normAij_s[i, ninds[j]]/normAij_p[i,nindp[j]] - 1.0) > 1e-4:
                if normAij_s[i, ninds[j]] < 1e-10 and normAij_p[i,nindp[j]] > 1e-10:
                    print("-- n: {0:.3e} {1:.3e} {2:.3e}".format(normAij_s[i, j], normAij_p[i, j], normAij_s[i, j]/normAij_p[i,j]))
    print("finished.")

    print("Checking Aij directions")
    for i in range(nneigh_s.shape[0]):
        ninds = neigh_s[i, :nneigh_s[i]].argsort()
        nindp = neigh_p[i, :nneigh_p[i]].argsort()

        pi = i+1
        
        for j in range(nneigh_s[i]):
            pj = neigh_s[ninds][j]
            if Aij_s[i, ninds[j], 0]/Aij_p[i, nindp[j], 0] < 0:
                print("-- x: {0:.3e} {1:.3e} {2:.3e}".format(Aij_s[i, j, 0], Aij_p[i, j, 0], Aij_s[i, j, 0]/Aij_p[i,j,0]))
            if Aij_s[i, ninds[j], 1]/Aij_p[i, nindp[j], 1] < 0:
                print("-- y: {0:.3e} {1:.3e} {2:.3e}".format(Aij_s[i, j, 1], Aij_p[i, j, 1], Aij_s[i, j, 1]/Aij_p[i,j,1]))

                #  print("-- n: {0:.3e} {1:.3e} {2:.3e}".format(normAij_s[i, j], normAij_p[i, j], normAij_s[i, j]/normAij_p[i,j]))

    print("finished.")







if __name__ == "__main__":
    snapfile = get_snapfile()

    print("working with", snapfile)

    Aij_swift, grad_swift, neighbours_swift, nneighs_swift = read_swift_Aij_data(snapfile)
    x, y, rho, m, h, ids = read_swift_particle_data(snapfile)


    # get kernel support radius instead of smoothing length
    H = ml.get_H(h)
    L = ml.read_boxsize(snapfile)
    Aij_python, neighbours_python, nneighs_python = ml.Aij_Ivanova_all(x, y, H, L=L, periodic=periodic, kernel=kernel)
    neighbour_ids_python = neighbours_python + 1

    check_neighbours(neighbours_swift, nneighs_swift, neighbour_ids_python, nneighs_python, x, y, h, L)
    compare_Aijs(Aij_swift, neighbours_swift, nneighs_swift, Aij_python, neighbours_python, nneighs_python)



    



def announce():

    print("CHECKING SURFACES.")
    print("tolerance is set to:", tolerance)
    print("NULL_RELATIVE is set to:", NULL_RELATIVE)
    print("---------------------------------------------------------")
    print()

    return
