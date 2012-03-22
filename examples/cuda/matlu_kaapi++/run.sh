#!/bin/bash

SCRATCH=/scratch/jvlima
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$SCRATCH/adonis/xkaapi/default/lib:$LD_LIBRARY_PATH

#niter=30
niter=1

function run_test {
    export KAAPI_CPUSET="4,5"
#    export KAAPI_CPUSET="4,5,10,11"
    export KAAPI_GPUSET="0~0,1~1"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"
#    msizes="2048"
    msizes="4096"
    bsizes="512"
    niter=1
    verif=1
    for m in $msizes ; do
	    for b in $bsizes; do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matlu_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=260046848 ./matlu_kaapi++ $m $b 1 $verif 
#	    KAAPI_STACKSIZE=260046848 gdb ./matlu_kaapi++
	    done
    done
}

run_test

exit 0

function run_getrf {
    ncpu=6
    ngpu="$1"
    gpuset="$2"

    out="$SCRATCH/res/xkaapi-sgetrf-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET='0:5'
    msizes="1024 2048"
    bsizes="512"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matlu_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=260046848 ./matlu_kaapi++ $m $b 1 $verif 
	    #	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
    return
    bsizes="1024"
    msizes="$(seq 3072 1024 40960)"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matlu_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=260046848 ./matlu_kaapi++ $m $b 1 $verif 
    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

GPUSET='3~7 4~8 5~9 6~10 7~11'
init_gpuset="0~6"
let 'ngpu=1'
run_getrf "$ngpu" "$init_gpuset"
for gset in $GPUSET
do
    init_gpuset="$init_gpuset,$gset"
    let 'ngpu++'
    run_getrf "$ngpu" "$init_gpuset"
done

