#!/usr/bin/env sh

PAPI_DIR=$HOME/install/lib
KAAPI_DIR=$HOME/install/xkaapi_rose
CPUCOUNT=`getconf _NPROCESSORS_ONLN`
CPUMAX=$((CPUCOUNT - 1))

LD_LIBRARY_PATH=$KAAPI_DIR/lib:$PAPI_DIR \
    KAAPI_STACKSIZE=268435456 \
    KAAPI_CPUSET=`seq -s, 0 1 $CPUMAX` \
    KAAPI_WSSELECT="hierarchical" \
    ./main ;
