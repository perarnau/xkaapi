#!/bin/bash

export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

version="$(date +%s)"
dorun="yes"

function run_test {
#    export KAAPI_CPUSET="0,5,6,11"
#    export KAAPI_GPUSET="0~1,1~2,2~3,3~4,4~7,5~8,6~9,7~10"
    export KAAPI_CPUSET="0,1"
    export KAAPI_GPUSET="0~2"

#    export KAAPI_CPUSET="0,1,5,6,10,11"
#    export KAAPI_GPUSET="1~2,2~3,3~4,4~7,5~8,6~9"

#    export KAAPI_CPUSET="0:11"

#    export KAAPI_CPUSET="4"
#    export KAAPI_CPUSET="0:3"
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

#    msizes="40960"
#    msizes="10240"
    msizes="4096"
#    msizes="20480"
#    msizes="16384"
    bsizes="300"
#    bsizes="512 1024"
    niter=1
#    verif=1
    export KAAPI_WINDOW_SIZE=2
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matlu_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE_MASTER=536870912 ./matlu_kaapi++ $m $b 1 $verif 
#	    KAAPI_STACKSIZE_MASTER=536870912 gdb ./matlu_kaapi++ 
	    done
	done
    done
}

run_test
exit 0
