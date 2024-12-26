#!/usr/bin/env python3

# Generate some simple IC to test concatGadgetHdf5ForVR.py
# using pNbody.
# will create a nfiles non-overlapping files,
# where the content is displaced in the x direction.


import pNbody as pn
from pNbody import ic
from numpy import random as r
import numpy as np


nfiles = 8
nparts = 10000
for i in range(nfiles):
    nb = ic.box(nparts, 0.5, 0.5, 0.5, ftype='gh5')
    fname = 'output.'+str(i).zfill(2)+'.gh5'
    nb.rename(fname)
    nb.translate([i+0.5, 0.5, 0.5])
    nb.boxsize = nfiles

    nb.write()
    #  nb.display(size=(nfiles,nfiles),shape=(256,256),palette='light')
    #
    #  nb.info()
