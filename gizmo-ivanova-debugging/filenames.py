#!/usr/bin/env python3

# storing global filenames


from my_utils import yesno, one_arg_present
import astro_meshless_surfaces as ml


snap = "0001"  # which snap to use; default value


def get_srcfile():
    """
    Generate name for source file to read in of swift gizmo debug dump
    """
    global snap
    fname_prefix = "swift-gizmo-debug-dump_"

    # read in cmd line arg snapshot number if present and convert it to formatted string
    snap = ml.snapstr(one_arg_present(snap))

    srcfile = fname_prefix + snap + ".dat"
    print("Working with file", srcfile)
    return srcfile


def get_dumpfiles():
    """
    Generate pickle dump filenames
    """

    global snap
    snap = ml.snapstr(one_arg_present(snap))
    print("Working with snapshot", snap)

    swift_dump = "gizmo-debug-swift-data_" + snap + ".pkl"
    part_dump = "gizmo-debug-swift-particle-data_" + snap + ".pkl"
    python_surface_dump = "gizmo-debug-python-surface-data_" + snap + ".pkl"
    python_grad_dump = "gizmo-debug-python-gradient-data_" + snap + ".pkl"

    return swift_dump, part_dump, python_surface_dump, python_grad_dump
