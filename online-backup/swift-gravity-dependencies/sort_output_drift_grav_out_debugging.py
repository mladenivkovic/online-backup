#!/usr/bin/env python3

# ========================================================================
# Check which cell is adding tasks in one output and not in other
# Works for hand-added output in engine_maketasks()
# You need to remove everything from the logs except the debugging output
# you added yourself from the logs
#
# Usage:
#   sort_output_drift_grav_out_debugging.py output_1.log output_2.log
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


log1cells = np.loadtxt(log1, dtype=int, delimiter=" ", usecols=[5])
log2cells = np.loadtxt(log2, dtype=int, delimiter=" ", usecols=[5])


in_common = 0
is_in_log2 = np.zeros(log1cells.shape, dtype=bool)
is_in_log1 = np.zeros(log2cells.shape, dtype=bool)

for i, cell in enumerate(log1cells):
    check = log2cells == cell
    if check.any():
        in_common += 1
        is_in_log2[i] = True
        is_in_log1[check] = True


print("In common:", in_common, "/", max(log1cells.shape[0], log2cells.shape[0]))
print(
    "In",
    log1,
    "but not",
    log2,
    ":",
    log1cells[~is_in_log2].shape[0],
    "/",
    log1cells.shape[0],
)
print(log1cells[~is_in_log2])
print(
    "In",
    log2,
    "but not",
    log1,
    ":",
    log2cells[~is_in_log1].shape[0],
    "/",
    log2cells.shape[0],
)
print(log2cells[~is_in_log1])
