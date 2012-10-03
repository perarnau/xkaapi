#!/bin/bash

#SCRATCH=$SCRATCH
#CUDADIR=$SCRATCH/install/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

version="$(date +%s)"
dorun="yes"

function run_test {
    export KAAPI_CPUSET="0:1"
#    export KAAPI_CPUSET="4"
#    export KAAPI_CPUSET="4,5,10,11"
#    export KAAPI_GPUSET="0~0"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3"
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

#    export KAAPI_DISPLAY_PERF=1
#
#    export KAAPI_PUSH_AFFINITY="writer"
#    export KAAPI_STEAL_AFFINITY="writer"
#    export KAAPI_PUSH_AFFINITY="locality"
#    export KAAPI_STEAL_AFFINITY="locality"


#    msizes="10240"
#    msizes="1024"
    msizes="2048"
#    msizes="16384"
    bsizes="512"
#    bsizes="1024"
    niter=1
    verif=1
    export KAAPI_WINDOW_SIZE=2
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matlu_right-looking_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 ./matlu_right-looking_kaapi++ $m $b 1 $verif 
#	    KAAPI_STACKSIZE=536870912 gdb ./matlu_nopiv_kaapi++ 
	    done
	done
    done
}

run_test
exit 0
