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

        self.RTStarIact = None
        self.RTCallsIactGradient = None
        self.RTCallsIactTransport = None

        self.InjectionDone = None
        self.ThermochemistryDone = None
        self.TransportDone = None
        self.GradientsDone = None

        self.RadiationReceivedTot = None

        #  self.RTCalls_this_step = None
        #  self.RTTotalCalls = None

        #  self.neighbours_grad = None
        #  self.neighcells_grad = None
        #  self.nneigh_grad = None
        #  self.neighbours_transport = None
        #  self.neighcells_transport = None
        #  self.nneigh_transport = None
        #  self.this_cell = None
 
        #  self.hydro_neighbours_grad = None
        #  self.hydro_neighcells_grad = None
        #  self.hydro_nneigh_grad = None
        #  self.hydro_neighbours_transport = None
        #  self.hydro_neighcells_transport = None
        #  self.hydro_nneigh_transport = None
        #  self.hydro_this_cell = None

        #  self.h_grad = None
        #  self.h_transport = None
        #  self.h_hydro_grad = None
        #  self.h_force = None

        return


class RTStarData(object):
    """
    Object to store RT star particle data of a snapshot
    """

    def __init__(self):

        self.IDs = None
        self.coords = None
        self.h = None

        self.RTHydroIact = None
        self.EmissionRateSet = None

        self.RadiationEmittedTot = None

        #  self.RTCalls_this_step = None
        #  self.RTTotalCalls = None

        return


class RTSnapData(object):
    """
    Object to store RT snapshot data
    """

    def __init__(self):
        self.snapnr = None
        self.ncells = None
        self.boxsize = None
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
        newsnap.boxsize = F["Header"].attrs["BoxSize"]
        newsnap.ncells = F["Cells"]
        Gas = F['PartType0']
        ids = Gas["ParticleIDs"][:]
        inds = np.argsort(ids)

        newsnap.gas.IDs = ids[inds]
        newsnap.gas.coords = Gas["Coordinates"][:][inds]
        newsnap.gas.h = Gas["SmoothingLengths"][:][inds]

        newsnap.gas.RTStarIact = Gas["RTStarIact"][:][inds]
        newsnap.gas.RTCallsIactGradient = Gas["RTCallsIactGradient"][:][inds]
        newsnap.gas.RTCallsIactTransport = Gas["RTCallsIactTransport"][:][inds]
        newsnap.gas.InjectionDone = Gas["RTInjectionDone"][:][inds]
        newsnap.gas.GradientsDone = Gas["RTGradientsDone"][:][inds]
        newsnap.gas.TransportDone = Gas["RTTransportDone"][:][inds]
        newsnap.gas.ThermochemistryDone = Gas["RTThermochemistryDone"][:][inds]

        newsnap.gas.RadiationReceivedTot = Gas["RTRadReceivedTot"][:][inds]


        #------------------------
        # deprecated debugging
        #------------------------

        #  newsnap.gas.RTCalls_pair_injection = Gas["RTCallsPairInjection"][:][inds]
        #  newsnap.gas.RTCalls_self_injection = Gas["RTCallsSelfInjection"][:][inds]
        #  newsnap.gas.RTCalls_this_step = Gas["RTCallsThisStep"][:][inds]
        #  newsnap.gas.RTTotalCalls = Gas["RTTotalCalls"][:][inds]

        #  newsnap.gas.RTCallsIactGradientSym = Gas["RTCallsIactGradientSym"][:][inds]
        #  newsnap.gas.RTCallsIactGradientNonSym = Gas["RTCallsIactGradientNonSym"][:][inds]
        #  newsnap.gas.RTCallsIactTransportSym = Gas["RTCallsIactTransportSym"][:][inds]
        #  newsnap.gas.RTCallsIactTransportNonSym = Gas["RTCallsIactTransportNonSym"][:][inds]
        #  newsnap.gas.neighbours_grad = Gas["RTNeighsIactGrad"][:][inds]
        #  newsnap.gas.neighcells_grad = Gas["RTNeighCellsIactGrad"][:][inds]
        #  newsnap.gas.nneigh_grad = Gas["RTNrNeighIactGrad"][:][inds]
        #  newsnap.gas.neighbours_transport = Gas["RTNeighsIactTransport"][:][inds]
        #  newsnap.gas.neighcells_transport = Gas["RTNeighCellsIactTransport"][:][inds]
        #  newsnap.gas.nneigh_transport = Gas["RTNrNeighIactTransport"][:][inds]
        #  newsnap.gas.this_cell_grad = Gas["RTThisCellGrad"][:][inds]
        #  newsnap.gas.this_cell_transport = Gas["RTThisCellTransport"][:][inds]
        #
        #  newsnap.gas.hydro_neighbours_grad = Gas["RTHydroNeighsIactGrad"][:][inds]
        #  newsnap.gas.hydro_neighcells_grad = Gas["RTHydroNeighCellsIactGrad"][:][inds]
        #  newsnap.gas.hydro_nneigh_grad = Gas["RTHydroNrNeighIactGrad"][:][inds]
        #  newsnap.gas.hydro_neighbours_transport = Gas["RTHydroNeighsIactTransport"][:][inds]
        #  newsnap.gas.hydro_neighcells_transport = Gas["RTHydroNeighCellsIactTransport"][:][inds]
        #  newsnap.gas.hydro_nneigh_transport = Gas["RTHydroNrNeighIactTransport"][:][inds]
        #  newsnap.gas.hydro_this_cell_grad = Gas["RTHydroThisCellGrad"][:][inds]
        #  newsnap.gas.hydro_this_cell_transport = Gas["RTHydroThisCellTransport"][:][inds]
        #
        #  newsnap.gas.RTHydroCallsIactGradient = Gas["RTHydroCallsIactGradient"][:][inds]
        #  newsnap.gas.RTHydroCallsIactForce = Gas["RTHydroCallsIactForce"][:][inds]
        #  newsnap.gas.RTHydroCallsIactGradientSym = Gas["RTHydroCallsIactGradientSym"][:][inds]
        #  newsnap.gas.RTHydroCallsIactGradientNonSym = Gas["RTHydroCallsIactGradientNonSym"][:][inds]
        #  newsnap.gas.RTHydroCallsIactForceSym = Gas["RTHydroCallsIactForceSym"][:][inds]
        #  newsnap.gas.RTHydroCallsIactForceNonSym = Gas["RTHydroCallsIactForceNonSym"][:][inds]
        #
        #  newsnap.gas.h_grad = Gas["RTSmlGrad"][:][inds]
        #  newsnap.gas.h_transport = Gas["RTSmlTransport"][:][inds]
        #  newsnap.gas.h_hydro_grad = Gas["RTHydroSmlGrad"][:][inds]
        #  newsnap.gas.h_force = Gas["RTHydroSmlForce"][:][inds]


        Stars = F['PartType4']
        ids = Stars["ParticleIDs"][:]
        inds = np.argsort(ids)

        newsnap.stars.IDs = ids[inds]
        newsnap.stars.coords = Stars["Coordinates"][:][inds]
        newsnap.stars.h = Stars["SmoothingLengths"][:][inds]

        newsnap.stars.RTHydroIact = Stars["RTHydroIact"][:][inds]
        newsnap.stars.EmissionRateSet = Stars["RTEmissionRateSet"][:][inds]

        newsnap.stars.RadiationEmittedTot = Stars["RTRadEmittedTot"][:][inds]

        #------------------------
        # Deprecated debugging
        #------------------------

        #  newsnap.stars.RTCalls_this_step = Stars["RTCallsThisStep"][:][inds]
        #  newsnap.stars.RTTotalCalls = Stars["RTTotalCalls"][:][inds]

        snapdata.append(newsnap)

    return snapdata



