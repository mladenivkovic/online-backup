#!/bin/bash

#---------------------------
# re-run all examples.
#---------------------------

# whether to re-run swift to generate snapshots
# rerun_swift=true
rerun_swift=false

# whether you're doing a test run to see whether your code is not crashing,
# or you want quality pics
# testing=true
testing=false


dir=$PWD

#-------------------------
function test {
#-------------------------
    # Crash this script if python script or run script crashes
    CODE=$1
    if [[ $CODE -ne 0 ]]; then
	echo $1 " Exit Code " $CODE
	echo "Run failed"
	exit $1
    fi
}



#-------------------------
function execscript {
#-------------------------

    which=$1

    cd $which;
        
        echo
        echo ======================================================
        echo working for `basename $PWD`
        echo ======================================================

        if [ $rerun_swift = true ]; then
            ./run.sh
            test $?
        fi


        if [ -f makeIC.py ]; then
            pyfile=./visualise*.py
        else
            pyfile=./*.py
        fi

        if [ $testing = true ]; then
            echo overwriting nx for testing.
            sed -i 's/^nx = [0-9]*/nx = 10/' $pyfile
        else
            echo overwriting nx to nx = 100.
            sed -i 's/^nx = [0-9]*/nx = 100/' $pyfile
        fi
            

        $pyfile
        test $?

    cd $dir
}



#-------------------------
# Back that shit up son
#-------------------------

if [ ! -d backup ]; then
    mkdir -p backup
    cp results/*.png backup
fi





#-------------------------------------
# Get array of dirs where to work
#-------------------------------------

declare -a dirlist

dirlist+=(A_of_x_different_kernels_uniform)
# dirlist+=(A_of_x_for_fixed_particles)
# dirlist+=(A_of_x_perturbed_uniform)
dirlist+=(A_of_x_varying_neighbour_numbers)
dirlist+=(A_of_x_varying_neighbour_numbers_and_kernels)
dirlist+=(A_of_x_varying_neighbour_numbers_properly_perturbed)
dirlist+=(A_of_x_varying_neighbour_numbers_properly_uniform)
# dirlist+=(hopkins_vs_ivanova_arrows)
# dirlist+=(hopkins_vs_ivanova_arrows_perturbed)
# dirlist+=(uniform_arrows)

# if [ $testing = true ]; then
#     dirlist+=(hopkins_vs_ivanova_part_displaced_uniform/testing)
#     dirlist+=(part_displaced_uniform/testing)
#     dirlist+=(part_perturbed_uniform/testing)
# else
#     dirlist+=(hopkins_vs_ivanova_part_displaced_uniform/ics_and_outputs)
#     dirlist+=(part_displaced_uniform/ics_and_outputs)
#     dirlist+=(part_perturbed_uniform/ics_and_outputs)
# fi




# Directories not to include because there are special cases
# dirlist+=(check_volume)
# dirlist+=(three_particles_volume_distribution)
# dirlist+=(two_particles_primitive)
# dirlist+=(voronoi)




for d in "${dirlist[@]}"; do
    execscript $d;
done
