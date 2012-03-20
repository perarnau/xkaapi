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
    export KAAPI_CPUSET="0:1"
    export KAAPI_GPUSET="0~4"

    export COMPUTE_PROFILE=1
    export COMPUTE_PROFILE_CSV=1
    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"
#    msizes="4096"
    msizes="2048"
#    msizes="16384"
#    msizes="2048"
    bsizes="512"
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

run_test
exit 0

function run_sgemm {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    out="$SCRATCH/res/xkaapi-sgemm-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET="$cpuset"
    msizes="1024 2048 4096"
    bsizes="512"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET ./matprod_iter_kaapi++ $m $b $verif"
	    #	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
    bsizes="1024"
    msizes="$(seq 8192 2048 40960)"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET ./matprod_iter_kaapi++ $m $b $verif"
#	    KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

ncpu=1
cpuset="0"

ngpu=1
gpuset="0~4"
run_sgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=2
gpuset="0~4,1~5"
run_sgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=4
gpuset="0~4,1~5,2~6,3~7"
run_sgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=8
gpuset="0~4,1~5,2~6,3~7,4~8,5~9,6~10,7~11"
run_sgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"
