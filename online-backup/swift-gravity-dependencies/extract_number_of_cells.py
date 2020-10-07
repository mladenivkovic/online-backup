#!/usr/bin/env python3


import h5py
import os
import numpy as np


fulldir = os.listdir()

filelist = []
for f in fulldir:
    if f.startswith("store-") and f.endswith(".hdf5"):
        filelist.append(f)
filelist.sort()


ncells = np.zeros(len(filelist), dtype=int)
ncells_with_stars = np.zeros(len(filelist), dtype=int)
outputnr = np.zeros(len(filelist), dtype=int)
run = np.zeros(len(filelist), dtype=int)
totstars = np.zeros(len(filelist), dtype=int)

for i, f in enumerate(filelist):
    runstr = f[6:-10]
    outputnrstr = f[-9:-5]
    run[i] = int(runstr)
    outputnr[i] = int(outputnrstr)

    F = h5py.File(f, "r")
    C = F["Cells"]
    counts = C["Counts"]["PartType4"]
    ncells[i] = counts.shape[0]
    hasstars = counts.value > 0
    totstars[i] = np.sum(counts)
    ncells_with_stars[i] = counts[hasstars].shape[0]
    F.close()


header = "{0:8}|{1:8}|{2:8}|{3:17}|{4:8}".format(
    "run", "snap", "ncells", "cells with stars", "stars_tot"
)


def printline(header):
    for _ in header:
        print("-", end="")
    print("")


print(header)
printline(header)

for i in range(len(filelist)):
    if run[i - 1] < run[i]:
        printline(header)
    print(
        "{0:8d}|{1:8d}|{2:8d}|{3:17}|{4:8}".format(
            run[i], outputnr[i], ncells[i], ncells_with_stars[i], totstars[i]
        )
    )
