#!/usr/bin/env python3

#----------------------------------------------
# Collection of checks for the 'debug' RT
# scheme in swift.
#
# Usage:
#   ./rt-sanity-checks.py
#
# the script expects all the files to be called
# 'output_XXXX.hdf5', and be in this dir
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
#  print_diffs = False      # print differences you find
#  break_on_diff = True   # quit when you find a difference
break_on_diff = False   # quit when you find a difference

kernel_gamma = 1.825742



def check_hydro_sanity(snapdata):
    """
    Sanity checks for hydro variables.
    - injection always done?
    - gradients always done?
    - thermochemistry always done?
    - RT transport calls >= RT gradient calls?
    """

    nsnaps = len(snapdata)
    npart = snapdata[0].gas.coords.shape[0]

    #----------------------------------------------
    # check relative changes between two snapshots
    #----------------------------------------------
    for s in range(1, nsnaps):

        this = snapdata[s].gas
        prev = snapdata[s-1].gas

        print("Checking hydro sanity pt1",
            snapdata[s].snapnr, "->", snapdata[s-1].snapnr)

        # check number increase for total calls
        if (this.RTTotalCalls < prev.RTTotalCalls).any():
            print("--- Total Calls not consistent")
            if print_diffs:
                for i in range(npart):
                    if this.RTTotalCalls[i] < prev.RTTotalCalls[i]:
                        print("-----", this.RTTotalCalls[i], prev.RTTotalCalls[i])

            if break_on_diff:
                quit()




    #----------------------------------------------
    # check absolute values of every snapshot
    #----------------------------------------------
    for snap in snapdata:

        gas = snap.gas

        print("Checking hydro sanity pt2; snapshot", snap.snapnr)

        # --------------------------------------------------------------
        # check that photons have been updated (ghost1 called)
        # --------------------------------------------------------------

        mask = gas.InjectionDone != 1
        if mask.any():
            print("--- Some photons have injection finished != 1")
            if print_diffs:
                print("----- IDs with photons_updated==0:")
                print(gas.IDs[mask], gas.InjectionDone[mask])

            if break_on_diff:
                quit()

        # --------------------------------------------------------------
        # check that Gradient is finished
        # --------------------------------------------------------------
        mask = gas.GradientsDone == 1
        if mask.any():
            print("--- Some gradients were finalized != 1")
            if print_diffs:
                print("----- IDs with gradients done != 1:")
                print(gas.IDs[mask], gas.GradientsDone[mask])

            if break_on_diff:
                quit()

        # --------------------------------------------------------------
        # check that transport is finished
        # --------------------------------------------------------------
        mask =gas.TransportDone != 1
        if mask.any():
            print("--- Some transport was finalised != 1")
            if print_diffs:
                print("----- IDs with transport done != 1:")
                print(gas.IDs[mask], gas.TransportDone[mask])

            if break_on_diff:
                quit()


        # --------------------------------------------------------------
        # check that thermochemistry is finished
        # --------------------------------------------------------------
        mask = gas.ThermochemistryDone != 1
        if mask.any():
            print("--- Some thermochemistry done != 1")
            if print_diffs:
                print("----- IDs with Thermochemistry_done != 1:")
                print(gas.IDs[mask], gas.ThermochemistryDone[mask])

            if break_on_diff:
                quit()




        # --------------------------------------------------------------
        # check that number of calls to gradient interactions is
        # at least the number of calls to transport interactions
        # in RT interactions
        # --------------------------------------------------------------
        if (gas.RTCallsIactTransport < gas.RTCallsIactGradient).any():
            print("   Found RT transport calls < gradient calls:", 
                    np.count_nonzero(gas.RTCallsIactTransport < gas.RTCallsIactGradient), 
                    "/", npart)




    return


def check_stars_sanity(snapdata):
    """
    Sanity checks for stars variables.
    - total calls keep increasing?
    """

    nsnaps = len(snapdata)
    npart = snapdata[0].stars.coords.shape[0]

    #-------------------------------
    # check relative changes
    #-------------------------------
    for s in range(1, nsnaps):

        this = snapdata[s].stars
        prev = snapdata[s-1].stars

        print("Checking stars sanity pt1", snapdata[s].snapnr, '->', snapdata[s-1].snapnr)

        #  check number increase for total calls
        if (this.RTTotalCalls < prev.RTTotalCalls).any():
            print("--- Total Calls not consistent")
            if print_diffs:
                for i in range(npart):
                    if this.RTTotalCalls[i] < prev.RTTotalCalls[i]:
                        print("-----", this.RTTotalCalls[i], prev.RTTotalCalls[i])

            if break_on_diff:
                quit()



    #----------------------------------------------
    #  check consistency of individual snapshots
    #----------------------------------------------
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

            print("--- count", count, "/", this.EmissionRateSet.shape[0])

            if break_on_diff:
                quit()


    return





def deprecated_hydro_checks(snapdata):
    """
    Deprecated sanity/debugging checks.
    Kept here for quick copy-paste.
    """

    nsnaps = len(snapdata)
    npart = snapdata[0].gas.coords.shape[0]


    # check absolute values every snapshot
    for snap in snapdata:
        gas = snap.gas

        # --------------------------------------------------------------
        # is cell ID for gradient tasks in hydro/RT the same?
        # hydro <-> RT comparison
        # --------------------------------------------------------------
        if (gas.hydro_this_cell_grad != gas.this_cell_grad).any():
            print("--- Got different cell IDs for this_cell_grad")

            for i in range(npart):
                if gas.hydro_this_cell_grad[i] != gas.this_cell_grad[i]:
                    print("----- ID: {0:6d} RT: {1:18d} Hydro: {2:18d}".format(
                            gas.IDs[i], gas.this_cell_grad[i], gas.hydro_this_cell_grad[i]))
            if break_on_diff:
                quit()


        # --------------------------------------------------------------
        # is cell ID for force/transport tasks in hydro/RT the same?
        # hydro <-> RT comparison
        # --------------------------------------------------------------
        if (gas.hydro_this_cell_transport != gas.this_cell_transport).any():
            print("--- Got different cell IDs for this_cell_transport")

            for i in range(npart):
                if gas.hydro_this_cell_transport[i] != gas.this_cell_transport[i]:
                    print("----- ID: {0:6d} RT: {1:18d} Hydro: {2:18d}".format(
                            gas.IDs[i], gas.this_cell_transport[i], gas.hydro_this_cell_transport[i]))
            if break_on_diff:
                quit()


        # --------------------------------------------------------------
        # is cell ID for gradient/force tasks in hydro the same?
        # hydro <-> hydro comparison
        # --------------------------------------------------------------
        if (gas.hydro_this_cell_grad != gas.hydro_this_cell_transport).any():
            print("--- Got different cell IDs for grad/force in hydro")

            for i in range(npart):
                if gas.hydro_this_cell_grad[i] != gas.hydro_this_cell_transport[i]:
                    print("----- ID: {0:6d} Grad: {1:18d} Force: {2:18d}".format(
                            gas.IDs[i], gas.hydro_this_cell_grad[i], gas.hydro_this_cell_transport[i]))
            if break_on_diff:
                quit()

        # --------------------------------------------------------------
        # is cell ID for gradient/transport tasks in hydro the same?
        # RT <-> RT comparison
        # --------------------------------------------------------------
        if (gas.this_cell_grad != gas.this_cell_transport).any():
            print("--- Got different cell IDs for grad/transport in RT")

            for i in range(npart):
                if gas.this_cell_grad[i] != gas.this_cell_transport[i]:
                    print("----- ID: {0:6d} Grad: {1:18d} Transport: {2:18d}".format(
                            gas.IDs[i], gas.this_cell_grad[i], gas.this_cell_transport[i]))
            if break_on_diff:
                quit()


        # --------------------------------------------------------------
        # check number of neighbours for gradient interactions
        # hydro <-> RT comparison
        # --------------------------------------------------------------
        if (gas.hydro_nneigh_grad != gas.nneigh_grad).any():
            print("   Got different neighbour numbers for nneigh_grad between hydro and RT")
            print("     RT > Hydro:", np.count_nonzero(gas.hydro_nneigh_grad < gas.nneigh_grad), "/", npart)
            print("     RT < Hydro:", np.count_nonzero(gas.hydro_nneigh_grad > gas.nneigh_grad), "/", npart)
            different_cells = 0
            in_hydro_but_not_rt = 0
            in_rt_but_not_hydro = 0
            for i in range(npart):
                if gas.hydro_nneigh_grad[i] != gas.nneigh_grad[i]:
                    #  print("    ID: {0:6d} RT: {1:4d} Hydro: {2:4d}".format(
                    #          gas.IDs[i], gas.nneigh_grad[i], gas.hydro_nneigh_grad[i]))
                    cells_hydro = []
                    for j in range(gas.hydro_nneigh_grad[i]):
                        pjind = gas.hydro_neighbours_grad[i][j] - 1
                        if (gas.IDs[pjind] != gas.hydro_neighbours_grad[i][j]):
                            raise ValueError("ID[pjind] != neighbour_ID")
                        cellID = gas.hydro_this_cell_grad[pjind]
                        if cellID not in cells_hydro:
                            cells_hydro.append(cellID)

                    cells_RT = []
                    for j in range(gas.nneigh_grad[i]):
                        pjind = gas.neighbours_grad[i][j] - 1
                        if (gas.IDs[pjind] != gas.neighbours_grad[i][j]):
                            raise ValueError("ID[pjind] != neighbour_ID")
                        cellID = gas.this_cell_grad[pjind]
                        if cellID not in cells_RT:
                            cells_hydro.append(cellID)

                    diff = set(cells_hydro).symmetric_difference(set(cells_RT))
                    if len(diff) > 0:
                        different_cells += 1
                        RTnotinhydro = set(cells_RT).difference(set(cells_hydro))
                        if len(RTnotinhydro) > 0:
                            in_rt_but_not_hydro += 1
                        hydronotinRT = set(cells_hydro).difference(set(cells_RT))
                        if len(hydronotinRT) > 0:
                            in_hydro_but_not_rt += 1
                        #  print("      Particle {0:5d}: Different cells interacted.".format(gas.IDs[i]))
                        #  print("        In RT but not in hydro:", RTnotinhydro)
                        #  print("        In hydro but not in RT:", hydronotinRT)

            if different_cells > 0:
                print("     Found {0:5d} particles with different cell interactions".format(different_cells))
                print("        Parts with RT cells that are not in hydro: {0:5d}".format(in_rt_but_not_hydro))
                print("        Parts with hydro cells that are not in RT: {0:5d}".format(in_hydro_but_not_rt))
                    
            if break_on_diff:
                quit()
        else:
            print("   nneigh_grad is equal between hydro and RT :)")

        # --------------------------------------------------------------
        # check number of neighbours for gradient interactions
        # hydro <-> RT comparison
        # --------------------------------------------------------------
        if (gas.hydro_nneigh_transport != gas.nneigh_transport).any():
            print("   Got different neighbour numbers for nneigh_transport between hydro and RT")
            #  for i in range(npart):
            #      if gas.hydro_nneigh_transport[i] != gas.nneigh_transport[i]:
            #          #  print("    ID: {0:6d} RT: {1:4d} Hydro: {2:4d} RT transport: {3:4d} Hydro transport: {4:4d}".format(
            #          #           gas.IDs[i], gas.nneigh_transport[i], gas.hydro_nneigh_transport[i], gas.nneigh_grad[i], gas.hydro_nneigh_grad[i]))
            #          print("    ID: {0:6d} RT: {1:4d} Hydro: {2:4d}".format(
            #                  gas.IDs[i], gas.nneigh_transport[i], gas.hydro_nneigh_transport[i]))
            print("     RT > Hydro:", np.count_nonzero(gas.hydro_nneigh_transport < gas.nneigh_transport), "/", npart)
            print("     RT < Hydro:", np.count_nonzero(gas.hydro_nneigh_transport > gas.nneigh_transport), "/", npart)
            if break_on_diff:
                quit()
        else:
            print("   nneigh_transport is equal between hydro and RT :)")


        # --------------------------------------------------------------
        # check that number of calls to gradient interactions is
        # at least the number of calls to transport interactions
        # in RT interactions
        # Just the check is kept in the sanity checks, but the print
        # out details are deprecated
        # --------------------------------------------------------------
        if (gas.RTCallsIactTransport < gas.RTCallsIactGradient).any():
            print("   Found RT transport calls < gradient calls:", np.count_nonzero(gas.RTCallsIactTransport < gas.RTCallsIactGradient), "/", npart)

            print("   Found RT transport calls < gradient calls")
            for i in range(npart):
                if gas.RTCallsIactTransport[i] < gas.RTCallsIactGradient[i]:
                    print("   Particle ID {0:6d}".format(gas.IDs[i]))
                    print("      cell ID {0:18d}".format(gas.this_cell_transport[i]))
                    print("      calls grad: {0:4d} calls transport: {1:4d}".format(
                            gas.RTCallsIactGradient[i], gas.RTCallsIactTransport[i]))

                    ngb_grad = gas.neighbours_grad[i, :gas.nneigh_grad[i]]
                    ncell_grad = gas.neighcells_grad[i, :gas.nneigh_grad[i]]

                    ngb_transport = gas.neighbours_transport[i,:gas.nneigh_transport[i]]
                    ncell_transport = gas.neighcells_transport[i,:gas.nneigh_transport[i]]

                    for g, gid in enumerate(ngb_grad):
                        if gid not in ngb_transport:
                            # get compact supports
                            Hi = gas.h[i] * kernel_gamma
                            xi = gas.coords[i]
                            indj = gas.IDs == gid
                            Hj = gas.h[indj].item() * kernel_gamma
                            xj = gas.coords[indj].ravel()

                            # get r and periodicity corrections
                            r = 0.
                            for d in range(3):
                                dx = xi[d] - xj[d]
                                if dx < - snap.boxsize[d] * 0.5:
                                    dx += snap.boxsize[d]
                                if dx > snap.boxsize[d] * 0.5:
                                    dx -= snap.boxsize[d]
                                r += dx**2

                            r = np.sqrt(r)

                            print("      G!inT: ID {0:6d} cellID {1:18d} | r/Hi: {2:.3f} r/Hj: {3:.3f}".format(gid, ncell_grad[g], r/Hi, r/Hj))

                    for t, tid in enumerate(ngb_transport):
                        if tid not in ngb_grad:
                            # get compact supports
                            Hi = gas.h[i] * kernel_gamma
                            xi = gas.coords[i]
                            indj = gas.IDs == tid
                            Hj = gas.h[indj].item() * kernel_gamma
                            xj = gas.coords[indj].ravel()

                            # get r and periodicity corrections
                            r = 0.
                            for d in range(3):
                                dx = xi[d] - xj[d]
                                if dx < - snap.boxsize[d] * 0.5:
                                    dx += snap.boxsize[d]
                                if dx > snap.boxsize[d] * 0.5:
                                    dx -= snap.boxsize[d]
                                r += dx**2

                            r = np.sqrt(r)
                            print("      T!inG: ID {0:6d} cellID {1:18d} | r/Hi: {2:.3f} r/Hj: {3:.3f}".format(tid, ncell_transport[t], r/Hi, r/Hj))

            if break_on_diff:
                quit()


        else:
            print("   RT Transport calls >= RT Gradient calls everywhere :)")


        # --------------------------------------------------------------
        # check that number of calls to gradient interactions is
        # at least the number of calls to force interactions
        # in hydro interactions
        # --------------------------------------------------------------
        if (gas.RTHydroCallsIactForce < gas.RTHydroCallsIactGradient).any():
            print("   Found hydro force calls < gradient calls:", np.count_nonzero(gas.RTHydroCallsIactForce < gas.RTHydroCallsIactGradient), "/", npart )
            for i in range(npart):
                if gas.RTHydroCallsIactForce[i] < gas.RTHydroCallsIactGradient[i]:
                    print("   Particle ID {0:6d}".format(gas.IDs[i]))
                    print("      cell ID {0:18d}".format(gas.this_cell_transport[i]))
                    print("      calls grad: {0:4d} calls force: {1:4d}".format(
                            gas.RTHydroCallsIactGradient[i], gas.RTHydroCallsIactForce[i]))

                    ngb_grad = gas.hydro_neighbours_grad[i, :gas.hydro_nneigh_grad[i]]
                    ncell_grad = gas.hydro_neighcells_grad[i, :gas.hydro_nneigh_grad[i]]

                    ngb_transport = gas.hydro_neighbours_transport[i,:gas.hydro_nneigh_transport[i]]
                    ncell_transport = gas.hydro_neighcells_transport[i,:gas.hydro_nneigh_transport[i]]

                    for g, gid in enumerate(ngb_grad):
                        if gid not in ngb_transport:
                            # get compact supports
                            Hi = gas.h[i] * kernel_gamma
                            xi = gas.coords[i]
                            indj = gas.IDs == gid
                            Hj = gas.h[indj].item() * kernel_gamma
                            xj = gas.coords[indj].ravel()

                            # get r and periodicity corrections
                            r = 0.
                            for d in range(3):
                                dx = xi[d] - xj[d]
                                if dx < - snap.boxsize[d] * 0.5:
                                    dx += snap.boxsize[d]
                                if dx > snap.boxsize[d] * 0.5:
                                    dx -= snap.boxsize[d]
                                r += dx**2

                            r = np.sqrt(r)

                            print("      G!inF: ID {0:6d} cellID {1:18d} | r/Hi: {2:.3f} r/Hj: {3:.3f}".format(gid, ncell_grad[g], r/Hi, r/Hj))

                    for t, tid in enumerate(ngb_transport):
                        if tid not in ngb_grad:
                            # get compact supports
                            Hi = gas.h[i] * kernel_gamma
                            xi = gas.coords[i]
                            indj = gas.IDs == tid
                            Hj = gas.h[indj].item() * kernel_gamma
                            xj = gas.coords[indj].ravel()

                            # get r and periodicity corrections
                            r = 0.
                            for d in range(3):
                                dx = xi[d] - xj[d]
                                if dx < - snap.boxsize[d] * 0.5:
                                    dx += snap.boxsize[d]
                                if dx > snap.boxsize[d] * 0.5:
                                    dx -= snap.boxsize[d]
                                r += dx**2

                            r = np.sqrt(r)
                            print("      F!inG: ID {0:6d} cellID {1:18d} | r/Hi: {2:.3f} r/Hj: {3:.3f}".format(tid, ncell_transport[t], r/Hi, r/Hj))

            if break_on_diff:
                quit()
        else:
            print("   Hydro Force calls >= Hydro Gradient calls everywhere :)")



        # --------------------------------------------------------------
        #  Compare Smoothing Lengths between grad and transport
        #  RT <-> RT comparison
        # --------------------------------------------------------------
        mask = gas.h_grad != gas.h_transport
        if (mask).any():
            ndiff = np.count_nonzero(mask)
            maxdiff = np.abs(gas.h_grad[mask] / gas.h_transport[mask] - 1.).max()
            print("   Got different smoothing lengths for particles between RT gradient and RT transport: ", ndiff, "/", npart)
            print("     max(| h_grad / h_transport - 1|) = {0:.6f}".format(maxdiff))
        else:
            print("   Smoothing lengths between  RT gradient and RT transport are the same :)")

        # --------------------------------------------------------------
        #  Compare Smoothing Lengths between grad and force
        #  hydro <-> hydro comparison
        # --------------------------------------------------------------
        if (gas.h_hydro_grad != gas.h_force).any():
            ndiff = np.count_nonzero(gas.h_grad != gas.h_transport)
            print("   Got different smoothing lengths for particles between hydro gradient and hydro force: ", ndiff, "/", npart)
        else:
            print("   Smoothing lengths between hydro gradient and hydro force are the same :)")

        # --------------------------------------------------------------
        #  Compare Smoothing Lengths between grad and transport
        #  hydro <-> RT comparison
        # --------------------------------------------------------------
        mask = gas.h_grad != gas.h_hydro_grad
        if (mask).any():
            ndiff = np.count_nonzero(gas.h_grad != gas.h_hydro_grad)
            maxdiff = np.abs(gas.h_grad[mask] / gas.h_hydro_grad[mask] - 1.).max()
            print("   Got different smoothing lengths for particles between RT gradient and hydro gradient: ", ndiff, "/", npart)
            print("     max(| h_grad / h_grad_hydro - 1|) = {0:.6f}".format(maxdiff))
        else:
            print("   Smoothing lengths between hydro gradient and RT gradient are the same :)")

        # --------------------------------------------------------------
        #  Compare Smoothing Lengths between grad and force
        #  hydro <-> RT comparison
        # --------------------------------------------------------------
        if (gas.h_transport != gas.h_force).any():
            ndiff = np.count_nonzero(gas.h_transport != gas.h_force)
            print("   Got different smoothing lengths for particles between RT transport and hydro force: ", ndiff, "/", npart)
        else:
            print("   Smoothing lengths between hydro force and RT transport are the same :)")



        # --------------------------------------------------------------
        # Check that number of symmetric + nonsymmetric interactions is
        # also the number of total calls you get
        # Here for RT
        # --------------------------------------------------------------
        GradExpect = gas.RTCallsIactGradientSym + gas.RTCallsIactGradientNonSym
        if (gas.RTCallsIactGradient != GradExpect).any():
            print("--- gradient number of calls wrong :(")

            if print_diffs:
                for i in range(npart):
                    if gas.RTCallsIactGradient[i] != GradExpect[i]:
                        print("-----", gas.IDs[i], gas.RTCallsIactGradient[i], GradExpect[i])

            if break_on_diff:
                quit()

        TransportExpect = gas.RTCallsIactTransportSym + gas.RTCallsIactTransportNonSym
        if (gas.RTCallsIactTransport != TransportExpect).any():
            print("--- transport number of calls wrong :(")

            if print_diffs:
                for i in range(npart):
                    if gas.RTCallsIactTransport[i] != TransportExpect[i]:
                        print("-----", gas.IDs[i], gas.RTCallsIactTransport[i], TransportExpect[i])

            if break_on_diff:
                quit()


        # --------------------------------------------------------------
        # Check that number of symmetric + nonsymmetric interactions is
        # also the number of total calls you get
        # Here for Hydro
        # --------------------------------------------------------------
        GradExpect = gas.RTHydroCallsIactGradientSym + gas.RTHydroCallsIactGradientNonSym
        if (gas.RTHydroCallsIactGradient != GradExpect).any():
            print("--- gradient number of calls wrong :(")

            if print_diffs:
                for i in range(npart):
                    if gas.RTHydroCallsIactGradient[i] != GradExpect[i]:
                        print("-----", gas.IDs[i], gas.RTHydroCallsIactGradient[i], GradExpect[i])

            if break_on_diff:
                quit()

        ForceExpect = gas.RTHydroCallsIactForceSym + gas.RTHydroCallsIactForceNonSym
        if (gas.RTHydroCallsIactForce != ForceExpect).any():
            print("--- transport number of calls wrong :(")

            if print_diffs:
                for i in range(npart):
                    if gas.RTHydroCallsIactForce[i] != ForceExpect[i]:
                        print("-----", gas.IDs[i], gas.RTHydroCallsIactForce[i], ForceExpect[i])

            if break_on_diff:
                quit()






def main():
    """
    Main function to run.
    """

    snapdata = get_snap_data(prefix="output", skip_snap_zero=skip_snap_zero, skip_last_snap=skip_last_snap)

    check_hydro_sanity(snapdata)

    check_stars_sanity(snapdata)



if __name__ == "__main__":
    main()
