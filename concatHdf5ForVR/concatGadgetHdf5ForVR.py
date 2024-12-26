#!/usr/bin/env python3


# concatenate any number of hdf5 gadget output files
# into 1 gadget file for VR
#
# usage: concatGadgetHdf5ForVR.py <filelist>

import os, sys
import h5py

outputfile = 'concatenated.hdf5'
hfile = None
header = None

allfiles = []

#=====================
def getfiles():
#=====================
    
    global allfiles

    arglist = sys.argv[1:]

    if (len(arglist)==0):
        print("Need at least 1 file to work")
        quit()

    for f in arglist:
        if (not os.path.exists(f)):
            print("Can't find file", f)
            print("Exiting")
            quit

    allfiles = arglist
    return



#=====================
def write_header():
#=====================

    global hfile, header
    
    hfile = h5py.File(outputfile, 'w')
    hfile.create_group("Header")
    header = hfile['Header']

    firstfile = h5py.File(allfiles[0], 'r')
    hdr2 = firstfile['Header']
    
    # copy Header attributes
    for key in list(hdr2.attrs):
        header.attrs[key] = hdr2.attrs[key]
    #
    #  header.attrs['BoxSize'] = hdr2.attrs['BoxSize']

    firstfile.close()




#=====================
def copy_positions():
#=====================

    global hfile

    # first find how many particle types we are dealing with
    firstfile = h5py.File(allfiles[0], 'r')
    
    istype = True
    ntype = 0
    ptypes = []
    for ntype in range(100):
        ptype = 'PartType'+str(ntype)
        if ptype in firstfile.keys():
            print('Found', ptype)
            ptypes.append(ptype)

    ntypes = len(ptypes)

    if (ntypes==0):
        print("Found no particle types, did something go wrong?")
        quit()


    # create a group and a dataset for each particle type
    grps = [None for i in range(ntypes)]
    dsets = [None for i in range(ntypes)]
    for i in range(ntypes):
        ptype = ptypes[i] 
        grps[i] = hfile.create_group(ptype)
        dsets[i] = grps[i].create_dataset('Coordinates', (0,3), maxshape=(None,3), chunks=(1000000, 3))
        

    firstfile.close()


    # get header attributes that need to be updated and reset them to 0
    numpart_this_file = header.attrs['NumPart_ThisFile']
    numpart_this_file[:] = 0
    numpart_total = header.attrs['NumPart_Total']
    numpart_total[:] = 0
    numpart_total_highword = header.attrs['NumPart_Total_HighWord']
    numpart_total_highword[:] = 0

    for f in allfiles:
        newfile = h5py.File(f, 'r')
        print("working on file", f)

        # update header
        hdr = newfile['Header']

        numpart_this_file += hdr.attrs['NumPart_ThisFile']
        numpart_total += hdr.attrs['NumPart_Total']
        numpart_total_highword += hdr.attrs['NumPart_Total_HighWord']


        # resize dataset, then copy data
        for i in range(ntypes):
            ptype = ptypes[i] 
            srcgrp = newfile[ptype]
            coords = srcgrp['Coordinates']

            ishape=dsets[i].shape
            cshape=coords.shape
            newshape=(ishape[0]+cshape[0], 3)
            dsets[i].resize(newshape)

            dsets[i][ishape[0]:newshape[0],:] = coords[:,:]

            hfile.flush()


    header.attrs['NumPart_ThisFile'] = numpart_this_file
    header.attrs['NumPart_Total'] = numpart_total
    header.attrs['NumPart_Total_HighWord'] = numpart_total_highword





#=====================
def main():
#=====================

    getfiles()
    write_header()
    copy_positions()


    return


if __name__ == "__main__":
    main()
