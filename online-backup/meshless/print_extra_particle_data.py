#!/usr/bin/env python3

# print extra added particle's (with ID=101) data


import h5py
import os
import sys
import numpy

if len(sys.argv) < 2:
    print("Need hdf5 file")
    quit()


fname = sys.argv[1]

f = h5py.File(fname)

grp = f["PartType0"]
coords = grp["Coordinates"][:]
ids = grp["ParticleIDs"][:]

index = numpy.where(ids == 101)[0]
print(index)
print(coords[index])
