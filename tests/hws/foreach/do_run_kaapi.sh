#!/usr/bin/env sh
CPU_SET="0:47";
#CPU_SET=0,1,2,3,24,25,26,27;
KAAPI_HWS_LEVELS=NUMA \
KAAPI_EMITSTEAL=hws \
KAAPI_CPUSET="$CPU_SET" \
LD_LIBRARY_PATH=$HOME/install/xkaapi_hws/lib \
./a.out.kaapi
