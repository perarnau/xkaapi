#!/usr/bin/env sh

# idkoiff best configuration, correctly mapped
# CPU_SET=`seq -s' ' 0 2 15`;
# CPU_SET=`seq -s' ' 0 15`;

# idfreeze
# CPU_SET="0 1 2 3 24 25 26 27";

#idkoiff
# CPU_SET="0:15";
CPU_SET=`seq -s',' 0 2 15`;

GOMP_CPU_AFFINITY="$CPU_SET" \
./a.out.omp



