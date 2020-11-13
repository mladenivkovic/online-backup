#!/usr/bin/env python3


import numpy as np
import sys
import os


try:
    csvfile = sys.argv[1]
except IndexError:
    print("I need a cell_hierarchy_*.csv file as cmdline arg.")
    quit()

if not os.path.exists(csvfile):
    raise ValueError("File", csvfile, "not found")




cellIDs, parents, depths = np.loadtxt(csvfile, dtype=np.int64, usecols=[0, 1, 16], skiprows=2, delimiter = ",", unpack=True)


# check that negative indexes are top levels
negatives = cellIDs < 0
if (depths[negatives] != 0).any():
    print("Got negative values for non-top cells")
    print(cellIDs[negatives][depths[negatives] != 0])
    print(depths[negatives][depths[negatives] != 0])


# check that non-negative indexes are not top levels
nonnegatives = cellIDs > 0
if (depths[nonnegatives] == 0).any():
    print("Got positive values for top cells")
    print(cellIDs[nonnegatives][depths[nonnegatives] != 0])
    print(depths[nonnegatives][depths[nonnegatives] != 0])


# check uniqueness
cellIDs.sort()
uniques, unique_counts = np.unique(cellIDs, return_counts=True)
if uniques.shape != cellIDs.shape:
    print("Found non-unique IDs")
    print(uniques.shape, cellIDs.shape)
    print(uniques[unique_counts>1])
