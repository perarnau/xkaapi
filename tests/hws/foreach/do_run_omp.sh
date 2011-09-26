#!/usr/bin/env bash

HOSTNAME=`hostname`
if [ "$HOSTNAME" == "idfreeze" ]; then
    CPU_SET="0 1 2 3 24 25 26 27";
    CPU_COUNT=8;
    CPU_SET="$CPU_SET 4 5 6 7 28 29 30 31";
    CPU_COUNT=16;
    CPU_SET="$CPU_SET 8 9 10 11 32 33 34 35";
    CPU_COUNT=24;
    CPU_SET="$CPU_SET 12 13 14 15 36 37 38 39";
    CPU_COUNT=32;
    CPU_SET="$CPU_SET 16 17 18 19 40 41 42 43";
    CPU_COUNT=40;
    CPU_SET="$CPU_SET 20 21 22 23 44 45 46 47";
    CPU_COUNT=48;

    CPU_SET=`seq -s' ' 0 47`;
    CPU_COUNT=48;

elif [ "$HOSTNAME" == "idkoiff" ]; then
    CPU_SET=`seq -s' ' 0 15`;
fi

OMP_NUM_THREADS=$CPU_COUNT GOMP_CPU_AFFINITY="$CPU_SET" \
./a.out.omp
