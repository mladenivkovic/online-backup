#!/bin/bash

for x in `seq 1 2 199`; do
    for y in `seq 1 2 199`; do
        # create IC's
        fname="displacedUniformPlane-`printf '%03d' $x`-`printf '%03d' $y`.hdf5"
        snapname="snapshot-`printf '%03d' $x`-`printf '%03d' $y`"

        if [ ! -f $fname ] ; then
            echo "creating IC"
            python makeIC.py 10 $x $y
            err=$?
            if [ $err -ne 0 ]; then
                exit $err;
            fi
        else
            echo "Found file."
        fi

        # replace IC filname and output basename in yml file
        sed -i "s/snapshot-[0-9a-z.-]*/$snapname/" displacedUniformPlane.yml
        sed -i "s/displacedUniformPlane-[0-9a-z.-]*/$fname/" displacedUniformPlane.yml

        # run swift
        swift-2d --hydro --threads=4 -n 0 displacedUniformPlane.yml 2>&1 | tee output.log
    done
done

# delete unnecessary files

rm -r restart energy.txt ./dependency_graph.csv output.log snapshot*0001.hdf5 snapshot*.xmf task_level.txt
rm timesteps_4.txt used_parameters.yml unused_parameters.yml
