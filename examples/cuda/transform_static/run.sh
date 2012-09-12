#!/usr/bin/env sh
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

#export KAAPI_CPUSET='0'
export KAAPI_CPUSET='0:2'
export KAAPI_GPUSET='0~3'
#export KAAPI_DUMP_GRAPH='1'

msizes="2048000"
#msizes="2048"
bsizes="256 512"

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

