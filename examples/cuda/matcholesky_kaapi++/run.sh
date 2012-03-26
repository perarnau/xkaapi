#!/bin/bash

SCRATCH=/scratch/jvlima
CUDADIR=$SCRATCH/install/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH

version="$(date +%s)"
out="$SCRATCH/res/xkaapi-spotrf-4cpu1gpu-${version}.txt"
niter=1

function run_test {
    export KAAPI_CPUSET="4"
#    export KAAPI_CPUSET="4,5,10,11"
    export KAAPI_GPUSET="0~0"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"

#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"
#    msizes="10240"
    msizes="4096"
#    bsizes="1024"
#    bsizes="2048"
    bsizes="1024"
    niter=1
#    verif=1
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matcholesky_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=260046848 ./matcholesky_kaapi++ $m $b 1 $verif 
	    #	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

run_test

exit 0

function run_potrf {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    out="$HOME/res/xkaapi-dpotrf-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET="$cpuset"
    bsizes="256 512 1024"
    msizes="$(seq 1024 1024 10240)"
    for m in $msizes
    do
	    for b in $bsizes
	    do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matcholesky_kaapi++ $m $b $verif >> $out"
	    KAAPI_STACKSIZE=536870912  ./matcholesky_kaapi++ $m $b 1 $verif >> $out
	    #	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

ncpu=1
cpuset="4"

ngpu=1
gpuset="0~0"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=2
gpuset="0~0,1~1"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=4
gpuset="0~0,1~1,2~2,3~3"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=8
gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"
