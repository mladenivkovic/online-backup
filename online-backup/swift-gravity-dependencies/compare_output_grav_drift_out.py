#!/usr/bin/env python3

# ========================================================================
# Compares the output to stdout for creation of tasks of type
# gravity drift_out
# You need to remove everything from the logs except the debugging output
# you added yourself from the logs
#
# Usage:
#   compare_output_grav_drift_out.py output_1.log output_2.log
# ========================================================================

import sys
import os
import numpy as np


log1 = sys.argv[1]
if not os.path.exists(log1):
    raise ValueError("Provided file", log1, "doesn't exist.")

log2 = sys.argv[2]
if not os.path.exists(log2):
    raise ValueError("Provided file", log2, "doesn't exist.")


log1cellloc = np.loadtxt(log1, dtype=np.float, delimiter=",", usecols=[1, 2, 3])
log1case, log1depth = np.loadtxt(
    log1, dtype=int, delimiter=",", usecols=[0, 4], unpack=True
)

log2cellloc = np.loadtxt(log2, dtype=float, delimiter=",", usecols=[1, 2, 3])
log2case, log2depth = np.loadtxt(
    log2, dtype=int, delimiter=",", usecols=[0, 4], unpack=True
)

# sort
#  inds1 = np.lexsort((log1cellloc[:,2], log1cellloc[:,1], log1cellloc[:,0]))
#  sort1 = np.lexsort((log1cellloc[:,2], log1cellloc[:,1], log1cellloc[:,0]))
#  sort2 = np.lexsort((log2cellloc[:,2], log2cellloc[:,1], log2cellloc[:,0]))

#  log1cellloc= log1cellloc[sort1].copy()
#  log1case = log1case[sort1]
#  log1depth = log1depth[sort1]
#  log2cellloc = log2cellloc[sort2]
#  log2case = log2case[sort2]
#  log2depth = log2depth[sort2].copy()


checked = np.zeros(log2cellloc.shape[0], dtype=int)
matched = 0
matched_case1 = 0
matched_case2 = 0
print("In", log1, "but not in", log2)
print(
    "{0:12} {1:12} {2:12} {3:12} | {4:12}".format(
        "loc[0]", "loc[1]", "loc[2]", "depth", "case"
    )
)
for i, loc1 in enumerate(log1cellloc):
    found = False
    for j, loc2 in enumerate(log2cellloc):
        #  if checked[j]:
        #      continue
        if log2depth[j] == log1depth[i]:
            if (loc1 == loc2).all():
                checked[j] = True
                found = True

                if log1case[i] != log2case[j]:
                    print("Oh no...")

                matched += 1
                if log1case[i] == 1:
                    matched_case1 += 1
                elif log1case[i] == 2:
                    matched_case2 += 1

                break

    if not found:
        print(
            "{0:12.6f} {1:12.6f} {2:12.6f} {3:12d} | {4:12d}".format(
                loc1[0], loc1[1], loc1[2], log1depth[i], log1case[i]
            )
        )


print("")
print("In", log2, "but not in", log1)
print(
    "{0:12} {1:12} {2:12} {3:12} | {4:12}".format(
        "loc[0]", "loc[1]", "loc[2]", "depth", "case"
    )
)

for i, c in enumerate(checked):
    if not c:
        print(
            "{0:12.6f} {1:12.6f} {2:12.6f} {3:12d} | {4:12d}".format(
                loc1[0], loc1[1], loc1[2], log1depth[i], log1case[i]
            )
        )

print()
print("Matched:", matched, "Case 1:", matched_case1, "Case 2:", matched_case2)
