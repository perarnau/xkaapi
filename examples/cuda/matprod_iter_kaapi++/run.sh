#!/bin/bash

#SCRATCH=/scratch/jvlima
CUDA_HOME=$SCRATCH/install/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH

niter=1
#niter=30
version="$(date +%s)"

#export KAAPI_DUMP_GRAPH='1'

function run_test {
#    export KAAPI_CPUSET="4"
    export KAAPI_CPUSET="0"
#    export KAAPI_CPUSET="4,5,10,11"
#    export KAAPI_GPUSET="0~0"
#   export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"

#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_DUMP_GRAPH=1
#    export KAAPI_DOT_NOVERSION_LINK=1
#    export KAAPI_DOT_NODATA_LINK=1
#    export KAAPI_DOT_NOLABEL_INFO=1
#    export KAAPI_DOT_NOACTIVATION_LINK=1
#    export KAAPI_DOT_NOLABEL=1

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"
    msizes="1024"
#    msizes="4096"
#    msizes="8192"
#    msizes="2048"
    bsizes="512"
#    nwindow="1 2 4 8 10 12 16"
    niter=1
    verif=1
#    for w in $nwindow
#    do
    export KAAPI_WINDOW_SIZE=2
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "window $KAAPI_WINDOW_SIZE $KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
#       KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
#    done	
}

run_test
exit 0
