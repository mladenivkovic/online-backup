#!/bin/bash

#========================================
# Script to quickly compile swift with
# the right flags and recompile as little
# as possible and necessary.
#
# use ./myinstall to just recompile using
# the previously used flags, provided the
# file .last_compile is present in this
# directory. Otherwise, it will just use
# the default flags below.
# You can also give it specific flags for
# which to compile. See below for
# possibilities (search keyword MYFLAGS)
#========================================

# change silent to false if you want the full ./configure and make outputs written
# to stdout and stderr. Otherwise, will be written to $logfile.
# script will exit if there was an error, so you'll know if something goes wrong.

silent=false
logfile=log_of_my_install
if [ -f $logfile ]; then rm $logfile; fi


DEBUGFLAGS=''           # will be overwritten by $DEBUGFLAGS_IF_IN_USE if you select debug option
# with optimization
DEBUGFLAGS_IF_IN_USE="  --enable-debug
                        --enable-sanitizer
                        --enable-undefined-sanitizer
                        --enable-debugging-checks"
                        # if debug is selected, these debugging flags will be used.
# without optimization
# DEBUGFLAGS_IF_IN_USE="  --enable-debug
#                         --enable-sanitizer
#                         --enable-optimization=no
#                         --enable-undefined-sanitizer
#                         --enable-debugging-checks
                        # --enable-task-debugging"
                        # if debug is selected, these debugging flags will be used.
DEFAULTFLAGS='          --enable-mpi=no 
                        --disable-doxygen-doc'
DIMFLAGS=''             # default 3D
# without Ivanova
# GIZMOFLAGS="            --with-hydro=gizmo-mfv
#                         --with-riemann-solver=hllc"
# with Ivanova
GIZMOFLAGS="            --with-hydro=gizmo-mfv 
                        --with-riemann-solver=hllc
                        --enable-ivanova-surfaces"
LIBFLAGS="              --with-parmetis 
                        --with-jemalloc" 
                        # --with-hdf5=$HDF5_ROOT/bin/h5pcc"

EXTRA_CFLAGS=""

ADDFLAGS="" # other additional flags






#======================================
# Function definitions
#======================================

function errexit() {
    # usage: errexit $? "optional message string"
    if [[ "$1" -ne 0 ]]; then
        echo "ERROR OCCURED. ERROR CODE $1"
        if [[ $# > 1 ]]; then
            echo "$2"
        fi
        traceback 1
        exit $1
    else
        return 0
    fi
}


function traceback
{
    # Hide the traceback() call.
    local -i start=$(( ${1:-0} + 1 ))
    local -i end=${#BASH_SOURCE[@]}
    local -i i=0
    local -i j=0

    echo "Traceback (last called is first):" 1>&2
    for ((i=start; i < end; i++)); do
      j=$(( i - 1 ))
      local function="${FUNCNAME[$i]}"
      local file="${BASH_SOURCE[$i]}"
      local line="${BASH_LINENO[$j]}"
      echo "     ${function}() in ${file}:${line}" 1>&2
    done
}




function file_separator() {
    # usage: filename "text to add"
    echo "=======================================================" >> $1
    echo "=======================================================" >> $1
    echo "========" $2 >> $1
    echo "=======================================================" >> $1
    echo "=======================================================" >> $1
    return 0
}










#======================================
# The party starts here
#======================================



if [ ! -f ./configure ]; then
    ./autogen.sh
fi








#--------------------------------------
# standardize comp flag
#--------------------------------------

# HERE ARE MYFLAGS
comp=default
dim=3d

if [[ $# == 0 ]]; then
    echo "NO ARGUMENTS GIVEN. COMPILING THE SAME WAY AS LAST TIME."
    comp=last
else

    while [[ $# > 0 ]]; do
    arg="$1"

    case $arg in

        default | -d | d | 3 | 3d | --3d | -3d)
            echo COMPILING DEFAULT 3D
            dim=3d
        ;;

        clean | c | -c | --c | --clean)
            echo COMPILING CLEAN
            GIZMOFLAGS=${GIZMOFLAGS//--enable-ivanova-surfaces/}
            clean='true'
        ;;

        1 | --1d | -1d | -1 | --1 | 1d)
            echo COMPILING 1D SWIFT
            DIMFLAGS='--with-hydro-dimension=1'
            dim=1d
        ;;

        2 | --2d | -2d | -2 | --2 | 2d)
            echo COMPILING 2D SWIFT
            DIMFLAGS='--with-hydro-dimension=2'
            dim=2d
        ;;

        last | --last | -l | --l)
            echo "COMPILING WITH SAME FLAGS AS LAST TIME"
            comp=last
        ;;
        
        me | my | mine | deb | debug | test)
            echo "ADDING DEBUG FLAGS"
            debug='true'
            DEBUGFLAGS=$DEBUGFLAGS_IF_IN_USE
        ;;


        wendland | w | w6 )
            echo "ADDING WENDLAND C6 FLAGS"
            used_kernel='true'
            kernelname='wendland-C6'
            ADDFLAGS="$ADDFLAGS"" --with-kernel=wendland-C6 --disable-vec --disable-hand-vec"
        ;;

        *)
            echo "COMPILING WITH SAME FLAGS AS LAST TIME BY WILDCARD"
            comp=last
        ;;

    esac
    shift
    done
fi






#---------------------------------
# generate exec filename suffix
#---------------------------------

program_suffix=-"$dim"

if [ "$clean" = 'true' ]; then
    program_suffix="$program_suffix""-clean"
fi
if [ "$debug" = 'true' ]; then
    program_suffix="$program_suffix""-debug"
fi
if [ "$used_kernel" = "true" ]; then
    program_suffix="$program_suffix"-"$kernelname"
fi

allflags="$LIBFLAGS ""$GIZMOFLAGS ""$DEFAULTFLAGS"" $ADDFLAGS"" $DEBUGFLAGS"" $DIMFLAGS"" CFLAGS=$EXTRA_CFLAGS" 










#--------------------------------------
# Check if reconfiguration is necessary
#--------------------------------------


reconfigure=false


if [ -f .last_compile ]; then
    last_comp=`head -n 1 .last_compile` # only first line!
    if [ $comp != "last" ]; then # if you don't just want to repeat the same compilation

        # check that you have identical flags
        # first if all in last_comp are present in allflags
        for flag in $last_comp; do
            if [[ "$allflags" != *"$flag"* ]]; then
                echo found unmatched flag $flag
                echo will reconfigure.
                reconfigure=true
                break
            fi
        done
        if [ "$reconfigure" != 'true' ]; then
            # now if all in allflags are present in last_comp
            for flag in $allflags; do
                if [[ "$last_comp" != *"$flag"* ]]; then
                    echo found unmatched flag $flag
                    echo will reconfigure.
                    reconfigure=true
                    break
                fi
            done
        fi
    else
        # if .last_compilation exists and comp is same as last, also read in the names
        allflags=$last_comp
        lastname=`sed -n 2p .last_compile`
        lastname_mpi=`sed -n 3p .last_compile`
    fi

else
    # if no .last_compile is present
    reconfigure='true'
    if [ "$comp" == 'last' ]; then
        lastname=swift
        lastname_mpi=swift_mpi
    fi
fi


# if it's comp_clean, assume you haven't been up this far,
# so reconfigure anyhow
# if [ "$comp_clean" = 'true' ]; then
#     reconfigure=true
# fi









#-------------------------------
# configure depending on case
#-------------------------------

if [ "$reconfigure" = "true" ]; then
    file_separator $logfile "make clean"
    if [ "$silent" = 'true' ]; then
        echo make clean
        make clean >> "$logfile"
        errexit $?
    else
        make clean | tee -a $logfile
        errexit $?
    fi

    echo configure flags are:
    for flag in $allflags; do
        echo "   " $flag
    done

    ./configure $allflags
    errexit $?

else
    echo skipping configure.
fi









#-------------------------------
# compile
#-------------------------------

if [ "$silent" = "true" ]; then
    echo making.
    file_separator $logfile "make"
    make -j >> $logfile
    errexit $?
else
    make -j | tee -a $logfile
    errexit "${PIPESTATUS[0]}"
fi









#--------------------------------------
# store what this compilation was
#--------------------------------------
echo "$allflags" | tr -d \\n | sed -r 's/\s+/ /g' > .last_compile
echo >> .last_compile # tr -d \\n removes all newlines, including the last one, so add one here










#-------------------------------
# rename executables
#-------------------------------

echo renaming.

if [ $comp != 'last' ]; then
    execname="./examples/swift""$program_suffix"
    execname_mpi="./examples/swift""$program_suffix"-mpi
else
    execname=$lastname
    execname_mpi=$lastname_mpi
fi






mv ./examples/swift "$execname"
echo "./examples/swift -> $execname"
echo finished $execname
if [ -f ./examples/swift_mpi ]; then 
    mv ./examples/swift_mpi "$execname_mpi"
    echo "./examples/swift_mpi -> $execname_mpi"
    echo finished $execname_mpi
fi

# store last used names
echo "$execname" >> .last_compile
echo "$execname_mpi" >> .last_compile
echo finished.
