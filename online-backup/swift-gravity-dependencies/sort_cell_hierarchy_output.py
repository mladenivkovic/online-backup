#!/usr/bin/env python3

# ========================================================================
# Sorts the cell_hierarchy_output by name.
#
# Usage:
#   sort_cell_hierarchy_output.py cell_hierarchy_XXXX_YYYY.csv outputfile
# ========================================================================

import sys
import os
import numpy as np


inputfile = sys.argv[1]
if not os.path.exists(inputfile):
    raise ValueError("Provided file", inputfile, "doesn't exist.")

outputfile = sys.argv[2]

f = open(inputfile)
all_lines = f.readlines()
f.close()


# ----------------------
# Sort by name
# ----------------------

#  names = [0 for line in all_lines]
#
#  for i, line in enumerate(all_lines):
#      if i == 0:
#          continue
#      comma = 0
#      c = line[comma]
#      while c != ',':
#          comma += 1
#          c = line[comma]
#          if comma == len(line) - 1:
#              raise IndexError("Reached end of the line. Line was:\n", line)
#      names[i] = int(line[:comma])
#
#  #  print("Before")
#  #  for i in range(1, 11):
#  #      print(names[i], all_lines[i][:6])
#  sorted_names, sorted_lines = zip(*sorted(zip(names[1:], all_lines[1:])))
#  #  print("After")
#  #  for i in range(1, 11):
#  #      print(sorted_names[i], sorted_lines[i][:6])
#
#
#  newfile = open(outputfile, 'w')
#  newfile.write(all_lines[0])
#  for line in sorted_lines:
#      newfile.write(line)
#  newfile.close()


# -------------------------
# Sort by position
# -------------------------

loc = np.loadtxt(inputfile, usecols=(8, 9, 10), skiprows=2, delimiter=",")
inds = np.lexsort((loc[:, 2], loc[:, 1], loc[:, 0]))

newfile = open(outputfile, "w")
newfile.write(all_lines[0])
newfile.write(all_lines[1])
for i in inds:
    # skip first two columns
    line = all_lines[i + 2]
    nfound = 0
    c = 0
    while nfound != 2:
        comma = line[c]
        if comma == ",":
            nfound += 1
        c += 1

    newfile.write(all_lines[i + 2][c:])
newfile.close()
