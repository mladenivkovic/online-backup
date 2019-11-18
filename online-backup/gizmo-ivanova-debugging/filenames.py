#!/usr/env/python3

# storing global filenames


from my_utils import yesno, one_arg_present
import meshless as ms


snap = '0001'                   # which snap to use; default value

#---------------------------------
def get_srcfile():
#---------------------------------
    """
    Generate name for source file to read in of swift gizmo debug dump
    """
    global snap
    fname_prefix = 'swift-gizmo-debug-dump_'

    # read in cmd line arg snapshot number if present and convert it to formatted string
    snap = ms.snapstr(one_arg_present(snap))

    srcfile = fname_prefix+snap+'.dat'
    return srcfile






#---------------------------
def get_dumpfiles():
#---------------------------
    """
    Generate pickle dump filenames
    """

    global snap
    snap = ms.snapstr(one_arg_present(snap))
 
    swift_dump = 'gizmo-debug-swift-data_'+snap+'.pkl'
    part_dump = 'gizmo-debug-swift-particle-data_'+snap+'.pkl'
    python_surface_dump = 'gizmo-debug-python-surface-data_'+snap+'.pkl'
    python_grad_dump = 'gizmo-debug-python-gradient-data_'+snap+'.pkl'

    return swift_dump, part_dump, python_surface_dump, python_grad_dump
