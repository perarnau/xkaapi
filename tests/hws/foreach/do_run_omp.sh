#!/usr/bin/env bash

HOSTNAME=`hostname`
if [ "$HOSTNAME" == "idfreeze" ]; then
    CPU_SET='0 1 2 3 24 25 26 27';
elif [ "$HOSTNAME" == "idkoiff" ]; then
    CPU_SET=`seq -s' ' 0 15`;
fi

GOMP_CPU_AFFINITY="$CPU_SET" \
numactl --interleave=all \
./a.out.omp



