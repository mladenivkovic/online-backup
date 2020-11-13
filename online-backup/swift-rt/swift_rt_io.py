#!/usr/bin/env python3

#-----------------------------------------
# Module containing RT I/O routines for
# debugging scheme
#-----------------------------------------


import os
import h5py
import numpy as np


class RTGasData(object):
    """
    Object to store RT gas particle data of a snapshot
    """

    def __init__(self):
        self.IDs = None
        self.coords = None
        self.h = None
        self.RTCallsIactGradient = None
        self.RTCallsIactGradientSym = None
        self.RTCallsIactGradientNonSym = None
        self.RTCallsIactTransport = None
        self.RTCalls_pair_injection = None
        self.RTCalls_self_injection = None
        self.RTCalls_this_step = None
        self.RTStarIact = None
        self.RTTotalCalls = None
        self.photons_updated = None
        # TODO: for later
        #  self.ThermochemistryDone = None
        #  self.TransportDone = None
        #  self.GradientsDone = None

        return


class RTStarData(object):
    """
    Object to store RT star particle data of a snapshot
    """

    def __init__(self):
        self.IDs = None
        self.coords = None
        self.h = None
        self.RTCalls_pair_injection = None
        self.RTCalls_self_injection = None
        self.RTCalls_this_step = None
        self.RTHydroIact = None
        self.RTTotalCalls = None
        self.EmissionRateSet = None
        return


class RTSnapData(object):
    """
    Object to store RT snapshot data
    """

    def __init__(self):
        self.snapnr = None
        self.ncells = None
        self.stars = RTStarData()
        self.gas = RTGasData()
        return


def get_snap_data(prefix="output_", skip_snap_zero=False, skip_last_snap=False):
    """
    Finds all prefix_XXXX.hdf5 files and reads
    the RT data in.

    Parameters
    ----------

    prefix: str
        file name prefix of snapshots. Don't include the
        underscore!

    skip_snap_zero: bool
        whether to skip the snapshot prefix_0000.hdf5

    skip_last_snap: bool
        whether to skip the last available snapshot


    Returns
    -------

    snapdata: list
        list of RTSnapData objects filled out with actual
        snapshot data
    """

    snapdata = []

    ls = os.listdir()
    hdf5files = []
    for f in ls:
        if f.startswith(prefix+"_") and f.endswith(".hdf5"):
            hdf5files.append(f)

    if len(hdf5files) == 0:
        raise IOError("No "+prefix+"_XXXX.hdf5 files found in this directory")

    hdf5files.sort()


    for f in hdf5files:
        snapnrstr = f[len(prefix)+1:len(prefix)+5]
        snapnr = int(snapnrstr)

        if skip_snap_zero and snapnr == 0:
            continue
        if skip_last_snap and f == hdf5files[-1]:
            continue

        newsnap = RTSnapData()
        newsnap.snapnr = snapnr

        F = h5py.File(f, 'r')
        newsnap.ncells = F["Cells"]
        Gas = F['PartType0']
        ids = Gas["ParticleIDs"][:]
        inds = np.argsort(ids)
        newsnap.gas.IDs = ids[inds]
        newsnap.gas.coords = Gas["Coordinates"][:][inds]
        newsnap.gas.h = Gas["SmoothingLengths"][:][inds]
        newsnap.gas.RTCalls_pair_injection = Gas["RTCallsPairInjection"][:][inds]
        newsnap.gas.RTCalls_self_injection = Gas["RTCallsSelfInjection"][:][inds]
        newsnap.gas.RTCalls_this_step = Gas["RTCallsThisStep"][:][inds]
        newsnap.gas.RTStarIact = Gas["RTStarIact"][:][inds]
        newsnap.gas.RTTotalCalls = Gas["RTTotalCalls"][:][inds]
        newsnap.gas.RTCallsIactGradient = Gas["RTCallsIactGradient"][:][inds]
        newsnap.gas.RTCallsIactTransport = Gas["RTCallsIactTransport"][:][inds]
        newsnap.gas.photons_updated = Gas["RTPhotonsUpdated"][:][inds]
        # TODO: for later
        #  newsnap.gas.GradientsDone = Gas["RTGradientsFinished"][:][inds]
        #  newsnap.gas.TransportDone = Gas["RTTransportDone"][:][inds]
        #  newsnap.gas.ThermochemistryDone = Gas["RTThermochemistryDone"][:][inds]
        newsnap.gas.RTCallsIactGradientSym = Gas["RTCallsIactGradientSym"][:][inds]
        newsnap.gas.RTCallsIactGradientNonSym = Gas["RTCallsIactGradientNonSym"][:][inds]
        newsnap.gas.RTCallsIactTransportSym = Gas["RTCallsIactTransportSym"][:][inds]
        newsnap.gas.RTCallsIactTransportNonSym = Gas["RTCallsIactTransportNonSym"][:][inds]



        Stars = F['PartType4']
        ids = Stars["ParticleIDs"][:]
        inds = np.argsort(ids)
        newsnap.stars.IDs = ids[inds]
        newsnap.stars.coords = Stars["Coordinates"][:][inds]
        newsnap.stars.h = Stars["SmoothingLengths"][:][inds]
        newsnap.stars.RTCalls_pair_injection = Stars["RTCallsPairInjection"][:][inds]
        newsnap.stars.RTCalls_self_injection = Stars["RTCallsSelfInjection"][:][inds]
        newsnap.stars.RTCalls_this_step = Stars["RTCallsThisStep"][:][inds]
        newsnap.stars.RTHydroIact = Stars["RTHydroIact"][:][inds]
        newsnap.stars.RTTotalCalls = Stars["RTTotalCalls"][:][inds]
        newsnap.stars.EmissionRateSet = Stars["RTEmissionRateSet"][:][inds]


        snapdata.append(newsnap)

    return snapdata



