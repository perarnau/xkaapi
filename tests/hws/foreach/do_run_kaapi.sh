#!/usr/bin/env bash

HOSTNAME=`hostname`
if [ "$HOSTNAME" == "idfreeze" ]; then
    CPU_SET=0,1,2,3,24,25,26,27;
    CPU_SET="$CPU_SET,4,5,6,7,28,29,30,31";
    CPU_SET="$CPU_SET,8,9,10,11,32,33,34,35";
    CPU_SET="$CPU_SET,12,13,14,15,36,37,38,39";
    CPU_SET=0:47
elif [ "$HOSTNAME" == "idkoiff" ]; then
    CPU_SET="0:15";
fi

KAAPI_HWS_LEVELS=NUMA \
KAAPI_EMITSTEAL=hws \
KAAPI_CPUSET="$CPU_SET" \
LD_LIBRARY_PATH=$HOME/install/xkaapi_hws/lib \
./a.out.kaapi
