#!/usr/bin/env sh

# idfreeze
#CPU_SET="0:47";
#CPU_SET=0,1,2,3,24,25,26,27;

#idkoiff
CPU_SET="0:15";
#CPU_SET=`seq -s',' 0 2 15`;

KAAPI_HWS_LEVELS=NUMA \
KAAPI_EMITSTEAL=hws \
KAAPI_CPUSET="$CPU_SET" \
LD_LIBRARY_PATH=$HOME/install/xkaapi_hws/lib \
./a.out.kaapi
