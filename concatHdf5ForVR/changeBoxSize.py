#!/usr/bin/env python3

#-------------------------------------------
# change the boxsize in the hdf5 file
# resulting from concatGadgetHdf5ForVR.py
# to be a 1d-array 
# (box being an arbitrary rectangle)
#
# usage: ./changeBoxSize.py
#-------------------------------------------


import h5py
import numpy as np

hdfile = 'concatenated.hdf5'

f = h5py.File(hdfile)

# get current boxsize
head = f['Header']
BS = head.attrs['BoxSize']

# now assume y and z axis have boxsize of 1
head.attrs['BoxSize'] = np.array([BS, 1, 1])

f.close()
