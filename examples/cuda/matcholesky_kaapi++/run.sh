#!/bin/bash

SCRATCH=$SCRATCH
CUDADIR=$SCRATCH/install/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/sched-prio-inf/lib:$LD_LIBRARY_PATH

version="$(date +%s)"

function run_test {
#    export KAAPI_CPUSET="0"
    export KAAPI_CPUSET="4"
#    export KAAPI_CPUSET="4,5,10,11"
    export KAAPI_GPUSET="0~0"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"

#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"

#    export KAAPI_DUMP_GRAPH=1
#    export KAAPI_DOT_NOVERSION_LINK=1
#    export KAAPI_DOT_NODATA_LINK=1
#    export KAAPI_DOT_NOLABEL_INFO=1
#    export KAAPI_DOT_NOACTIVATION_LINK=1
#    export KAAPI_DOT_NOLABEL=1

#    msizes="10240"
    msizes="256"
#    msizes="16384"
#    bsizes="2048"
    bsizes="128"
    niter=1
#    verif=1
    export KAAPI_WINDOW_SIZE=2
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matcholesky_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 ./matcholesky_kaapi++ $m $b 1 $verif 
#	    KAAPI_STACKSIZE=536870912 gdb ./matcholesky_kaapi++ 
	    done
	done
    done
}

run_test
exit 0
