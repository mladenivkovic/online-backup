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


import numpy as np
from swift_rt_io import get_snap_data


# some behaviour options
skip_snap_zero = True   # skip snap_0000.hdf5
skip_last_snap = True   # skip snap_0max.hdf5
skip_coords = False      # skip coordinates check
skip_sml = False      # skip smoothing lengths check
print_diffs = True      # print differences you find
#  break_on_diff = True   # quit when you find a difference
break_on_diff = False   # quit when you find a difference

# tolerance for a float to be equal
float_comparison_tolerance = 1e-3 # with feedback
#  float_comparison_tolerance = 1e-5 # without feedback



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

        # Smoothing Lengths
        if not skip_sml:

            diff = np.abs((ref.gas.h - compare.gas.h) / ref.gas.h)
            if (diff > float_comparison_tolerance).any():
                print("--- Coordinates vary")
                if print_diffs:
                    for i in range(npart):
                        if ((ref.gas.h[i] - compare.gas.h[i])/ref.gas.h[i]).any():
                            print(ref.gas.h[i], "|", compare.gas.h[i])

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

        # Photon number updates
        if (ref.gas.photons_updated != compare.gas.photons_updated).any():
            print("--- Calls to photons_updated")

            if print_diffs:
                for i in range(npart):
                    if ref.gas.photons_updated[i] != compare.gas.photons_updated[i]:
                        print("-----", ref.gas.IDs[i], ref.gas.photons_updated[i], compare.gas.photons_updated[i])

            if break_on_diff:
                quit()


        # Gradient Loop Calls
        if (ref.gas.RTCallsIactGradient != compare.gas.RTCallsIactGradient).any():
            print("--- Calls to iact gradient")

            if print_diffs:
                for i in range(npart):
                    if ref.gas.RTCallsIactGradient[i] != compare.gas.RTCallsIactGradient[i]:
                        print("-----", ref.gas.IDs[i], ref.gas.RTCallsIactGradient[i], compare.gas.RTCallsIactGradient[i])

            if break_on_diff:
                quit()

        # Transport Loop Calls
        if (ref.gas.RTCallsIactTransport != compare.gas.RTCallsIactTransport).any():
            print("--- Calls to iact gradient")

            if print_diffs:
                for i in range(npart):
                    if ref.gas.RTCallsIactTransport[i] != compare.gas.RTCallsIactTransport[i]:
                        print("-----", ref.gas.IDs[i], ref.gas.RTCallsIactTransport[i], compare.gas.RTCallsIactTransport[i])

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

        # check number increase for total calls
        totalCallsExpect = prev.RTTotalCalls + this.RTCalls_this_step
        if (this.RTTotalCalls != totalCallsExpect).any():
            print("--- Total Calls not consistent")
            if print_diffs:
                for i in range(npart):
                    if this.RTTotalCalls[i] != totalCallsExpect[i]:
                        print("-----", this.RTTotalCalls[i], prev.RTTotalCalls[i], prev.RTCalls_this_step[i])

            if break_on_diff:
                quit()





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

        # check that number of calls to gradient interactions is
        # same as number of calls to transport interactions
        if (gas.RTCallsIactTransport < gas.RTCallsIactGradient).any():
            print("transport calls < gradient calls")
            mask = gas.RTCallsIactTransport < gas.RTCallsIactGradient
            print(gas.IDs[mask], gas.RTCallsIactGradient[mask], gas.RTCallsIactTransport[mask])

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


        # Smoothing Lengths
        if not skip_sml:

            diff = np.abs((ref.gas.h - compare.gas.h) / ref.gas.h)
            if (diff > float_comparison_tolerance).any():
                print("--- Coordinates vary")
                if print_diffs:
                    for i in range(npart):
                        if ((ref.gas.h[i] - compare.gas.h[i])/ref.gas.h[i]).any():
                            print(ref.gas.h[i], "|", compare.gas.h[i])

                if break_on_diff:
                    quit()


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
    - emission rates set?
    """

    nsnaps = len(snapdata)
    npart = snapdata[0].stars.coords.shape[0]

    # check relative changes
    for s in range(1, nsnaps):

        this = snapdata[s].stars
        prev = snapdata[s-1].stars

        print("Checking stars sanity pt1", snapdata[s].snapnr, '->', snapdata[s-1].snapnr)

        #  check number increase for total calls
        totalCallsExpect = prev.RTTotalCalls + this.RTCalls_this_step
        if (this.RTTotalCalls != totalCallsExpect).any():
            print("--- Total Calls not consistent: decreasing?")
            if print_diffs:
                for i in range(npart):
                    if this.RTTotalCalls[i] != totalCallsExpect[i]:
                        print("-----", this.RTTotalCalls[i], prev.RTTotalCalls[i], this.RTCalls_this_step)

            if break_on_diff:
                quit()



    #  check consistency of individual snapshots
    for snap in snapdata:

        print("Checking stars sanity pt2", snap.snapnr)
        this = snap.stars

        
        if (this.EmissionRateSet != 1).any():
            print("--- Emisison Rates not consistent")
            count = 0
            for i in range(npart):
                if this.EmissionRateSet[i] != 1:
                    count += 1
                    if print_diffs:
                        print("-----", this.EmissionRateSet[i], "ID", this.IDs[i])
            print("----- count: ", count, "/", this.EmissionRateSet.shape[0])

            if break_on_diff:
                quit()

    return




def main():
    """
    Main function to run.
    """

    snapdata = get_snap_data(prefix="uniform-rt", skip_snap_zero=skip_snap_zero, skip_last_snap=skip_last_snap)

    check_all_hydro_is_equal(snapdata)
    check_hydro_sanity(snapdata)

    check_all_stars_is_equal(snapdata)
    check_stars_sanity(snapdata)



if __name__ == "__main__":
    main()
