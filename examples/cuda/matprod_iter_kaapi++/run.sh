#!/bin/bash

CUDA_HOME=/usr
SCRATCH=/scratch/jvlima
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH

niter=1
#niter=30
version="$(date +%s)"

#export KAAPI_DUMP_GRAPH='1'

function run_test {
    export KAAPI_CPUSET="4,5,10,11"
    export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"

#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"
    msizes="8192"
#    msizes="4096"
#    msizes="2048"
#    msizes="16384"
#    msizes="2048"
    bsizes="1024"
    niter=1
#    verif=1
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 \
		    ./matprod_iter_kaapi++ $m $b $verif
#	    	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
#       KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

#run_test
#exit 0

function run_dgemm {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    out="$HOME/res/xkaapi-dgemm-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET="$cpuset"
    bsizes="512"
    msizes="$(seq 1024 1024 20480)"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
		echo "$KAAPI_CPUSET($ncpu) $KAAPI_GPUSET($ngpu) ./matprod_iter_kaapi++ $m $b $verif"
	    # KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif >> $out
    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

ncpu=1
cpuset="4"

ngpu=1
gpuset="0~0"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=2
gpuset="0~0,1~1"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=4
gpuset="0~0,1~1,2~2,3~3"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=8
gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"
