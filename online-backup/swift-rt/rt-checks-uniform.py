#!/usr/bin/env python3

#----------------------------------------------
# Collection of checks for the 'debug' RT
# scheme in swift.
#
# Usage:
#   ./rt-checks-uniform.py
#
# the script expects all the files to be called
# 'uniform-rt_XXXX.hdf5', and be in this dir
# where you're running the script from
#----------------------------------------------



import os
import numpy as np
import h5py


# some behaviour options
skip_snap_zero = True   # skip snap_0000.hdf5
skip_coords = False      # skip coordinates check
print_diffs = True      # print differences you find
break_on_diff = True   # quit when you find a difference 
break_on_diff = False   # quit when you find a difference

float_comparison_tolerance = 1e-5




class RTGasData(object):
    """
    Object to store RT gas particle data of a snapshot
    """

    def __init__(self):
        self.IDs = None
        self.coords = None
        self.RTCalls_pair_injection = None
        self.RTCalls_self_injection = None
        self.RTCalls_this_step = None
        self.photons_updated = None
        self.RTStarIact = None
        self.RTTotalCalls = None
        return


class RTStarData(object):
    """
    Object to store RT star particle data of a snapshot
    """

    def __init__(self):
        self.IDs = None
        self.coords = None
        self.RTCalls_pair_injection = None
        self.RTCalls_self_injection = None
        self.RTCalls_this_step = None
        self.RTHydroIact = None
        self.RTTotalCalls = None
        return


class RTSnapData(object):
    """
    Object to store RT snapshot data
    """


    def __init__(self):
        self.snapnr = None
        self.stars = RTStarData()
        self.gas = RTGasData()
        return




def get_snap_data():
    """
    Finds all 'uniform-rt_XXXX.hdf5' files and reads
    the RT data in.

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
        if f.startswith("uniform-rt_") and f.endswith("hdf5"):
            hdf5files.append(f)

    if len(hdf5files) == 0:
        raise IOError("No uniform-rt_XXXX.hdf5 files found in this directory")

    hdf5files.sort()


    for f in hdf5files:
        snapnrstr = f[11:15]
        snapnr = int(snapnrstr)
        if snapnr == 0:
            continue

        newsnap = RTSnapData()
        newsnap.snapnr = snapnr

        F = h5py.File(f, 'r')
        Gas = F['PartType0']
        ids = Gas["ParticleIDs"][:]
        inds = np.argsort(ids)
        newsnap.gas.IDs = ids[inds]
        newsnap.gas.coords = Gas["Coordinates"][:][inds]
        newsnap.gas.RTCalls_pair_injection = Gas["RTCallsPairInjection"][:][inds]
        newsnap.gas.RTCalls_self_injection = Gas["RTCallsSelfInjection"][:][inds]
        newsnap.gas.RTCalls_this_step = Gas["RTCallsThisStep"][:][inds]
        newsnap.gas.photons_updated = Gas["RTPhotonsUpdated"][:][inds]
        newsnap.gas.RTStarIact = Gas["RTStarIact"][:][inds]
        newsnap.gas.RTTotalCalls = Gas["RTTotalCalls"][:][inds]



        Stars = F['PartType4']
        ids = Stars["ParticleIDs"][:]
        inds = np.argsort(ids)
        newsnap.stars.IDs = ids[inds]
        newsnap.stars.coords = Stars["Coordinates"][:][inds]
        newsnap.stars.RTCalls_pair_injection = Stars["RTCallsPairInjection"][:][inds]
        newsnap.stars.RTCalls_self_injection = Stars["RTCallsSelfInjection"][:][inds]
        newsnap.stars.RTCalls_this_step = Stars["RTCallsThisStep"][:][inds]
        newsnap.stars.RTHydroIact = Stars["RTHydroIact"][:][inds]
        newsnap.stars.RTTotalCalls = Stars["RTTotalCalls"][:][inds]


        snapdata.append(newsnap)

    return snapdata



def check_all_hydro_is_equal(snapdata):
    """
    Check that all the hydro quantities are equal in every snapshot
    (for the relevant quantities of course.)
    """

    ref = snapdata[0]
    npart = ref.gas.coords.shape[0]

    for compare in snapdata[1:]:
        print("Comparing hydro", ref.snapnr, "->", compare.snapnr)

        # Coordinates
        if not skip_coords:

            diff = np.abs((ref.gas.coords - compare.gas.coords) / ref.gas.coords)
            if (diff > float_comparison_tolerance).any():
                print("--- Coordinates vary")
                if print_diffs:
                    for i in range(ref.gas.coords.shape[0]):
                        if ((ref.gas.coords[i] - compare.gas.coords[i])/ref.gas.coords[i]).any():
                            print(ref.gas.coords[i], "|", compare.gas.coords[i])

                if break_on_diff:
                    quit()


        # Pair Injection
        if (ref.gas.RTCalls_pair_injection != compare.gas.RTCalls_pair_injection).any():
            print("--- Calls Pair Injection Vary")

            if print_diffs:
                for i in range(npart):
                    if ref.gas.RTCalls_pair_injection[i] != compare.gas.RTCalls_pair_injection[i]:
                        print("-----", ref.gas.IDs[i], ref.gas.RTCalls_pair_injection[i], compare.gas.RTCalls_pair_injection[i])

            if break_on_diff:
                quit()


        # Self Injection
        if (ref.gas.RTCalls_self_injection != compare.gas.RTCalls_self_injection).any():
            print("--- Calls Self Injection Vary")

            if print_diffs:
                for i in range(npart):
                    if ref.gas.RTCalls_self_injection[i] != compare.gas.RTCalls_self_injection[i]:
                        print("-----", ref.gas.IDs[i], ref.gas.RTCalls_self_injection[i], compare.gas.RTCalls_self_injection[i])

            if break_on_diff:
                quit()


        # Calls this step
        if (ref.gas.RTCalls_this_step != compare.gas.RTCalls_this_step).any():
            print("--- Calls this step Vary")

            if print_diffs:
                for i in range(npart):
                    if ref.gas.RTCalls_this_step[i] != compare.gas.RTCalls_this_step[i]:
                        print("-----", ref.gas.IDs[i], ref.gas.RTCalls_this_step[i], compare.gas.RTCalls_this_step[i])

            if break_on_diff:
                quit()


        # Calls to star interactions
        if (ref.gas.RTStarIact != compare.gas.RTStarIact).any():
            print("--- Calls to star interactions vary")

            if print_diffs:
                for i in range(npart):
                    if ref.gas.RTStarIact[i] != compare.gas.RTStarIact[i]:
                        print("-----", ref.gas.IDs[i], ref.gas.RTStarIact[i], compare.gas.RTStarIact[i])

            if break_on_diff:
                quit()

    return


def check_hydro_sanity(snapdata):
    """
    Sanity checks for hydro variables.
    - photons updated every time step?
    - total calls keep increasing?
    """

    nsnaps = len(snapdata)
    npart = snapdata[0].gas.coords.shape[0]

    # check relative changes
    for s in range(1, nsnaps):

        this = snapdata[s].gas
        prev = snapdata[s-1].gas

        print("Checking hydro sanity pt1", 
            snapdata[s].snapnr, "->", snapdata[s-1].snapnr)

        # TODO: getting errors here :/
        # check number increase for total calls
        #  totalCallsExpect = prev.RTTotalCalls + this.RTCalls_this_step
        #  if (this.RTTotalCalls != totalCallsExpect).any():
        #      print("--- Total Calls not consistent")
        #      if print_diffs:
        #          for i in range(npart):
        #              if this.RTTotalCalls[i] != totalCallsExpect[i]:
        #                  print("-----", this.RTTotalCalls[i], prev.RTTotalCalls[i], prev.RTCalls_this_step[i])





    # check absolute values every snapshot
    for snap in snapdata:

        gas = snap.gas

        print("Checking hydro sanity pt2", snap.snapnr)

        # check that photons have been updated (ghost1 called)
        if (gas.photons_updated == 0).any():
            print("--- Some photons haven't been updated")
            if print_diffs:
                print("----- IDs with photons_updated==0:")
                print(gas.IDs[this.photons_updated == 0])

            if break_on_diff:
                quit()


    return


def check_all_stars_is_equal(snapdata):
    """
    Check that all the star quantities are equal in every snapshot
    (for the relevant quantities of course.)
    """

    ref = snapdata[0]
    npart = ref.stars.coords.shape[0]

    for compare in snapdata[1:]:
        print("Comparing stars", ref.snapnr, "->", compare.snapnr)

        # Coordinates
        if not skip_coords:

            diff = np.abs((ref.stars.coords - compare.stars.coords) / ref.stars.coords)
            if (diff > float_comparison_tolerance).any():
                print("--- Coordinates vary")
                if print_diffs:
                    for i in range(ref.stars.coords.shape[0]):
                        if ((ref.stars.coords[i] - compare.stars.coords[i])/ref.stars.coords[i]).any():
                            print(ref.stars.coords[i], "|", compare.stars.coords[i])

                if break_on_diff:
                    quit()

        # TODO: getting errors here. CHECK!
        # Pair Injection
        if (ref.stars.RTCalls_pair_injection != compare.stars.RTCalls_pair_injection).any():
            print("--- Calls Pair Injection Vary")

            if print_diffs:
                for i in range(npart):
                    if ref.stars.RTCalls_pair_injection[i] != compare.stars.RTCalls_pair_injection[i]:
                        print("-----", ref.stars.IDs[i], ref.stars.RTCalls_pair_injection[i], compare.stars.RTCalls_pair_injection[i])

            if break_on_diff:
                quit()


        # Self Injection
        if (ref.stars.RTCalls_self_injection != compare.stars.RTCalls_self_injection).any():
            print("--- Calls Self Injection Vary")

            if print_diffs:
                for i in range(npart):
                    if ref.stars.RTCalls_self_injection[i] != compare.stars.RTCalls_self_injection[i]:
                        print("-----", ref.stars.IDs[i], ref.stars.RTCalls_self_injection[i], compare.stars.RTCalls_self_injection[i])

            if break_on_diff:
                quit()


        # Calls this step
        if (ref.stars.RTCalls_this_step != compare.stars.RTCalls_this_step).any():
            print("--- Calls this step Vary")

            if print_diffs:
                for i in range(npart):
                    if ref.stars.RTCalls_this_step[i] != compare.stars.RTCalls_this_step[i]:
                        print("-----", ref.stars.IDs[i], ref.stars.RTCalls_this_step[i], compare.stars.RTCalls_this_step[i])

            if break_on_diff:
                quit()


        # Calls to star interactions
        if (ref.stars.RTHydroIact != compare.stars.RTHydroIact).any():
            print("--- Calls to hydro interactions vary")

            if print_diffs:
                for i in range(npart):
                    if ref.stars.RTHydroIact[i] != compare.stars.RTHydroIact[i]:
                        print("-----", ref.stars.IDs[i], ref.stars.RTHydroIact[i], compare.stars.RTHydroIact[i])

            if break_on_diff:
                quit()

    return


def check_stars_sanity(snapdata):
    """
    Sanity checks for stars variables.
    - total calls keep increasing?
    """

    nsnaps = len(snapdata)
    npart = snapdata[0].gas.coords.shape[0]

    # check relative changes
    for s in range(nsnaps):

        this = snapdata[s].stars
        prev = snapdata[s-1].stars

        print("Checking stars sanity pt1", 
            snapdata[s].snapnr)

        # TODO: getting errors here :/
        #  check number increase for total calls
        #  totalCallsExpect = prev.RTTotalCalls + this.RTCalls_this_step
        #  if (this.RTTotalCalls != totalCallsExpect).any():
        #      print("--- Total Calls not consistent")
        #      if print_diffs:
        #          for i in range(npart):
        #              if this.RTTotalCalls[i] != totalCallsExpect[i]:
        #                  print("-----", this.RTTotalCalls[i], prev.RTTotalCalls[i], prev.RTCalls_this_step[i])
        #
        #      if break_on_diff:
        #          quit()



    # check absolute values every snapshot
    #  for snap in snapdata:
    #
    #      gas = snap.gas
    #
    #      print("Checking stars sanity pt2", snap.snapnr)
    #
    #      # check that photons have been updated (ghost1 called)
    #      if (gas.photons_updated == 0).any():
    #          print("--- Some photons haven't been updated")
    #          if print_diffs:
    #              print("----- IDs with photons_updated==0:")
    #              print(gas.IDs[this.photons_updated == 0])
    #
    #          if break_on_diff:
    #              quit()
    #

    return




def main():
    """
    Main function to run.
    """

    snapdata = get_snap_data()

    check_all_hydro_is_equal(snapdata)
    check_hydro_sanity(snapdata)

    check_all_stars_is_equal(snapdata)
    check_stars_sanity(snapdata)



if __name__ == "__main__":
    main()
