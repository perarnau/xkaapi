#!/bin/bash

export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

export KAAPI_CPUSET='0'
export KAAPI_GPUSET='0~1'

#msizes="2048000"
msizes="10240"
bsizes="512"

verif=1

for m in $msizes
do
    for b in $bsizes
    do
    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
	    ./transform_static $m $b $verif"
    KAAPI_STACKSIZE=536870912 ./transform_static $m $b $verif
#    KAAPI_STACKSIZE=260046848 gdb ./transform_static
    done
done

